import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from scipy.optimize import curve_fit
from simulation.analysis import orientation_acf
from simulation.theory import (
    passive_msd, active_msd, active_msd_short_time, active_msd_long_time,
    active_msd_exponent,
    effective_diffusion, rotational_relaxation_time,
    orientation_acf_theory,
)
from visualization.plotter import plot_trajectory, plot_msd

_HERE = Path(__file__).parent  # always save PNGs next to main.py

# ── Parameter mode ────────────────────────────────────────────────────────────
# USE_PHYSICAL = True  → D_T and D_R derived from particle radius via
#                        Stokes-Einstein-Debye (Eqs. 1 & 2 in Romanczuk et al.)
# USE_PHYSICAL = False → use D_T and D_R values set directly below
USE_PHYSICAL = True

# Physical particle parameters (active when USE_PHYSICAL = True)
RADIUS_UM = 6.5     # particle radius [µm]  — hPSC dissociated diameter 10-15 µm → R ≈ 5-8 µm (see docs/hPSC_parameters.md)
T_K       = 310.0  # temperature [K]  (37 °C, physiological)
ETA_PA_S  = 1e-3   # dynamic viscosity [Pa·s]  (cell culture medium, water-like)

# Direct diffusion coefficients (active when USE_PHYSICAL = False)
D_T = 0.22   # translational diffusion [µm²/s]
D_R = 0.16   # rotational diffusion [rad²/s]

if USE_PHYSICAL:
    _phys = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
    D_T, D_R = _phys.D_T, _phys.D_R

V           = 0.004   # self-propulsion speed [µm/s]  (= 14.4 µm/h; literature: 5-20 µm/h, see docs/hPSC_parameters.md)
DT_SIM      = 10.0    # time step [s]  (D_R·dt ≈ 0.006 ≪ 1; ~161 steps per τ_R)
N_STEPS     = 800     # for trajectory plots  (800 × 10 s = 8000 s ≈ 5 τ_R)
N_STEPS_MSD = 800     # for MSD and ACF  (covers ≥ 5 τ_R so ACF decays to < 1 %)
N_ENSEMBLE  = 100


def _run(particle_cls, params: SimParams) -> np.ndarray:
    return Simulator(particle_cls(params), params).run()


def compare_boundary_conditions(
    particle_cls,
    base_params: SimParams,
    box_size: float,
    n_ensemble: int = 30,
    n_steps_msd: int | None = None,
) -> plt.Figure:
    """Run particle_cls under all four boundary modes and compare trajectories + MSD.

    Args:
        particle_cls: PassiveBrownianParticle or ActiveBrownianParticle.
        base_params:  SimParams instance whose D_T, D_R, v, dt, n_steps are reused.
                      boundary and box_size in base_params are ignored.
        box_size:     half-width of the confinement box [µm].
        n_ensemble:   number of realizations for the MSD comparison.
        n_steps_msd:  steps to use for the MSD panel (defaults to base_params.n_steps).
                      Use a larger value to resolve the long-time saturation.

    Returns:
        Figure with 2 rows: top = 4 trajectory plots, bottom = MSD comparison.
    """
    modes = ["none", "reflect", "stop", "slip"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    is_passive = particle_cls is PassiveBrownianParticle
    particle_name = "Passive" if is_passive else f"Active (v={base_params.v:.1f} µm/s)"

    fig = plt.figure(figsize=(18, 9))
    traj_axes = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
    msd_ax = fig.add_subplot(2, 1, 2)

    def _params_for(mode, n_steps=None):
        return SimParams(
            D_T=base_params.D_T, D_R=base_params.D_R, v=base_params.v,
            dt=base_params.dt, n_steps=n_steps or base_params.n_steps,
            seed=base_params.seed,
            x0=base_params.x0, y0=base_params.y0, phi0=base_params.phi0,
            boundary=mode,
            box_size=box_size if mode != "none" else None,
        )

    for ax, mode, color in zip(traj_axes, modes, colors):
        traj = _run(particle_cls, _params_for(mode))
        ax.plot(traj[:, 0], traj[:, 1], lw=0.6, color=color, alpha=0.8)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="green", s=25, zorder=5)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="red",   s=25, zorder=5)
        if mode != "none":
            L = box_size
            rect = mpatches.FancyBboxPatch(
                (-L, -L), 2 * L, 2 * L,
                boxstyle="square,pad=0", fill=False,
                edgecolor="black", linewidth=1.5, linestyle="--",
            )
            ax.add_patch(rect)
            ax.set_xlim(-L * 1.15, L * 1.15)
            ax.set_ylim(-L * 1.15, L * 1.15)
        ax.set_aspect("equal")
        ax.set_title(f"boundary='{mode}'")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")

    dt = base_params.dt
    n_msd = n_steps_msd if n_steps_msd is not None else base_params.n_steps
    for mode, color in zip(modes, colors):
        trajs = [
            _run(particle_cls, SimParams(
                D_T=base_params.D_T, D_R=base_params.D_R, v=base_params.v,
                dt=dt, n_steps=n_msd, seed=s,
                x0=base_params.x0, y0=base_params.y0, phi0=base_params.phi0,
                boundary=mode,
                box_size=box_size if mode != "none" else None,
            ))
            for s in range(n_ensemble)
        ]
        t = np.arange(1, n_msd + 1) * dt
        msds = np.array([
            (tr[1:, 0] - tr[0, 0]) ** 2 + (tr[1:, 1] - tr[0, 1]) ** 2
            for tr in trajs
        ])
        msd_ax.loglog(t, msds.mean(axis=0), label=f"'{mode}'", color=color)

    # Free-space theory: valid at short times for all modes (before wall is reached)
    t_th = np.arange(1, n_msd + 1) * dt
    if is_passive:
        msd_th = passive_msd(t_th, base_params.D_T)
        theory_label = "free-space: 4·D_T·t  (short-time ref)"
    else:
        msd_th = active_msd(t_th, base_params.D_T, base_params.D_R, base_params.v)
        theory_label = "free-space theory  (short-time ref)"
    msd_ax.loglog(t_th, msd_th, color="gray", ls=":", lw=1.5, alpha=0.8,
                  label=theory_label)

    # Reflect long-time saturation: particle explores box uniformly → MSD → 2L²/3
    msd_sat = 2.0 * box_size**2 / 3.0
    msd_ax.axhline(msd_sat, color="black", ls="--", lw=0.8, alpha=0.5,
                   label=f"reflect saturation: 2L²/3 = {msd_sat:.1f} µm²")

    msd_ax.set_xlabel("time [s]")
    msd_ax.set_ylabel("MSD [µm²]")
    msd_ax.set_title(f"MSD — all boundary conditions  [{particle_name}]")
    msd_ax.legend(fontsize=8)
    msd_ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    return fig


def main():
    tau_R = rotational_relaxation_time(D_R)
    tau_T = 4.0 * D_T / V**2   # 4D_T·t = v²t² crossover (diffusive → ballistic)
    D_eff = effective_diffusion(D_T, D_R, V)
    l_p   = V / D_R             # persistence length = v × τ_R  [µm]
    t_max = N_STEPS_MSD * DT_SIM
    print(f"D_T = {D_T:.4f} µm²/s  |  D_R = {D_R:.2e} rad²/s")
    print(f"tau_T = {tau_T:.0f} s  |  tau_R = {tau_R:.0f} s  |  D_eff = {D_eff:.4f} µm²/s  |  l_p = {l_p:.1f} µm")
    print(f"t_max = {t_max:.0f} s  ({t_max/tau_R:.1f}× tau_R)")

    # ── 1. Single trajectory side-by-side ─────────────────────────────────────
    passive_params = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    active_params  = SimParams(D_T=D_T, D_R=D_R, v=V,   dt=DT_SIM, n_steps=N_STEPS, seed=42)

    traj_p = _run(PassiveBrownianParticle, passive_params)
    traj_a = _run(ActiveBrownianParticle,  active_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_trajectory(traj_p, title="Passive Brownian (Eq. 3)",             ax=axes[0])
    plot_trajectory(traj_a, title=f"Active Brownian v={V} µm/s (Eq. 4)", ax=axes[1])
    fig.tight_layout()
    fig.savefig(_HERE / "trajectories.png", dpi=150)
    plt.close(fig)
    print("Saved trajectories.png")

    # ── 2. Ensemble MSD + local exponent α(t) = d(log MSD)/d(log t) ──────────
    # Note: τ_T = 4D_T/v² >> τ_R for hPSCs (Pe < 1), so there is no ballistic
    # regime — the active MSD only shows a soft α peak slightly above 1 near τ_R.
    passive_trajs = [
        _run(PassiveBrownianParticle,
             SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS_MSD, seed=s))
        for s in range(N_ENSEMBLE)
    ]
    active_trajs = [
        _run(ActiveBrownianParticle,
             SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS_MSD, seed=s))
        for s in range(N_ENSEMBLE)
    ]

    t_th = np.arange(1, N_STEPS_MSD + 1) * DT_SIM

    # Ensemble-average MSDs
    msd_passive_sim = np.array([
        (tr[1:, 0] - tr[0, 0])**2 + (tr[1:, 1] - tr[0, 1])**2
        for tr in passive_trajs
    ]).mean(axis=0)
    msd_active_sim = np.array([
        (tr[1:, 0] - tr[0, 0])**2 + (tr[1:, 1] - tr[0, 1])**2
        for tr in active_trajs
    ]).mean(axis=0)

    # Theory MSDs and local exponents
    msd_passive_th = passive_msd(t_th, D_T)
    msd_active_th  = active_msd(t_th, D_T, D_R, V)
    alpha_active_th  = active_msd_exponent(t_th, D_T, D_R, V)
    alpha_passive_th = np.ones_like(t_th)

    # Peak of theory exponent
    peak_idx  = np.argmax(alpha_active_th)
    alpha_peak = alpha_active_th[peak_idx]
    t_peak     = t_th[peak_idx]

    Pe = V * RADIUS_UM / D_T   # Péclet number
    print(f"Pe = {Pe:.2f}  |  peak alpha = {alpha_peak:.4f}  at t = {t_peak:.0f} s  ({t_peak/tau_R:.2f} tau_R)")

    fig2, (ax_msd, ax_exp) = plt.subplots(2, 1, figsize=(9, 10), sharex=False)

    # ── Top: MSD ──────────────────────────────────────────────────────────────
    ax_msd.loglog(t_th, msd_passive_sim, color="tab:blue",  lw=1.5,
                  label=f"Passive sim  (N={N_ENSEMBLE})")
    ax_msd.loglog(t_th, msd_passive_th,  color="navy",     lw=1.5, ls="--", alpha=0.8,
                  label="4·D_T·t  (theory)")
    ax_msd.loglog(t_th, msd_active_sim,  color="tab:red",  lw=1.5,
                  label=f"Active sim  v={V} µm/s  (N={N_ENSEMBLE})")
    ax_msd.loglog(t_th, msd_active_th,   color="darkred",  lw=1.5, ls="--", alpha=0.8,
                  label="Exact theory")
    ax_msd.loglog(t_th, active_msd_long_time(t_th, D_T, D_R, V),
                  color="gray", ls="-.", lw=1.2,
                  label=f"Long-time: 4·D_eff·t  (D_eff/D_T = {D_eff/D_T:.2f})")
    ax_msd.axvline(tau_R, color="black",     ls="--", lw=1, alpha=0.5,
                   label=f"τ_R = {tau_R:.0f} s")
    ax_msd.axvline(tau_T, color="steelblue", ls=":",  lw=1, alpha=0.5,
                   label=f"τ_T = {tau_T:.0f} s  (> t_max: no ballistic regime, Pe={Pe:.2f})")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("Mean Squared Displacement — hPSC  (passive vs active)")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # Global power-law fit to active theory MSD over full window: MSD ~ t^alpha_fit
    alpha_global, log_c = np.polyfit(np.log(t_th), np.log(msd_active_th), 1)

    # ── Bottom: local exponent (theory only — sim derivative is too noisy) ────
    ax_exp.semilogx(t_th, alpha_passive_th, color="navy",    lw=2, ls="--",
                    label="Passive theory  (alpha = 1, always diffusive)")
    ax_exp.semilogx(t_th, alpha_active_th,  color="darkred", lw=2, ls="--",
                    label=f"Active theory  (instantaneous local exponent)")
    ax_exp.scatter([t_peak], [alpha_peak], color="darkred", s=80, zorder=5,
                   label=f"peak alpha = {alpha_peak:.4f}  at t = {t_peak:.0f} s = {t_peak/tau_R:.1f} tau_R")
    ax_exp.axhline(alpha_global, color="tab:green", ls="-.", lw=1.5,
                   label=f"global power-law fit over full window: alpha_fit = {alpha_global:.4f}")
    ax_exp.axhline(1, color="gray", ls=":",  lw=1, alpha=0.6, label="alpha = 1  (diffusive)")
    ax_exp.axhline(2, color="gray", ls="--", lw=1, alpha=0.6, label="alpha = 2  (ballistic, not reached here)")
    ax_exp.axvline(tau_R, color="black", ls="--", lw=1, alpha=0.5, label=f"tau_R = {tau_R:.0f} s")
    ax_exp.set_xlabel("time [s]")
    ax_exp.set_ylabel("local exponent  alpha(t)")
    ax_exp.set_title("MSD local exponent  alpha(t) = d(log MSD) / d(log t)\n"
                     "Simulation derivative omitted — numerical noise swamps the signal")
    ax_exp.set_ylim(0.8, 2.2)
    ax_exp.legend(fontsize=8)
    ax_exp.grid(True, which="both", ls="--", alpha=0.4)

    fig2.tight_layout()
    fig2.savefig(_HERE / "msd_comparison.png", dpi=150)
    plt.close(fig2)
    print("Saved msd_comparison.png")

    # ── 3. Boundary condition comparison — passive and active ─────────────────
    # Passive box L=10 µm: tau_cross ~ L²/(4·D_T) ≈ 714 s ≈ 71 steps → saturation visible.
    # Active  box L=15 µm: tau_cross ~ L/v = 3750 s ≈ 375 steps < t_max → particle hits walls.
    base_p = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    fig3p = compare_boundary_conditions(
        PassiveBrownianParticle, base_p, box_size=10.0, n_ensemble=100,
        n_steps_msd=N_STEPS_MSD,
    )
    fig3p.savefig(_HERE / "boundary_comparison_passive.png", dpi=150)
    plt.close(fig3p)
    print("Saved boundary_comparison_passive.png")

    base_a = SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    fig3a = compare_boundary_conditions(
        ActiveBrownianParticle, base_a, box_size=15.0, n_ensemble=100,
        n_steps_msd=N_STEPS_MSD,
    )
    fig3a.savefig(_HERE / "boundary_comparison_active.png", dpi=150)
    plt.close(fig3a)
    print("Saved boundary_comparison_active.png")

    # ── 4. Orientation ACF + correlation time + persistence length ────────────
    # max_lag = N_STEPS_MSD covers t_max ≈ 5 τ_R so the ACF decays to < 1 %.
    max_lag = N_STEPS_MSD
    dt_arr  = np.arange(max_lag) * DT_SIM

    _, acf_o = orientation_acf(active_trajs, max_lag=max_lag)

    # Nonlinear least-squares fit of ACF to exp(-D_R_fit * tau) in linear space.
    # Restrict to ACF > 0.2 (first ~1.6 τ_R): below this the noise (~6% of ACF)
    # dominates and pulls the slope.  The SNR is good in the 0.2–1.0 range.
    fit_mask = acf_o > 0.2
    (D_R_fit,), _ = curve_fit(
        lambda t, dr: np.exp(-dr * t),
        dt_arr[fit_mask], acf_o[fit_mask],
        p0=[D_R],
        bounds=(0, np.inf),
    )
    tau_c   = 1.0 / D_R_fit
    l_p_fit = V * tau_c          # persistence length from fitted τ_c
    print(f"Fitted D_R = {D_R_fit:.2e} rad²/s  (input = {D_R:.2e})   tau_c = {tau_c:.0f} s   l_p = {l_p_fit:.1f} µm")

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.plot(dt_arr, acf_o, color="tab:blue",
             label="simulation")
    ax4.plot(dt_arr, orientation_acf_theory(dt_arr, D_R), "--",
             color="tab:orange", alpha=0.9, lw=2,
             label=f"theory (input):  D_R = {D_R:.2e} rad²/s,  τ_R = {tau_R:.0f} s,  l_p = {l_p:.1f} µm")
    ax4.plot(dt_arr, orientation_acf_theory(dt_arr, D_R_fit), "-.",
             color="tab:green", alpha=0.9, lw=2,
             label=f"fit (recovered): D_R = {D_R_fit:.2e} rad²/s,  τ_c = {tau_c:.0f} s,  l_p = {l_p_fit:.1f} µm")
    ax4.axvline(tau_R, ls="--", color="tab:orange", alpha=0.5,
                label=f"τ_R = {tau_R:.0f} s  (theory)")
    ax4.axvline(tau_c, ls="-.", color="tab:green", alpha=0.5,
                label=f"τ_c = {tau_c:.0f} s  (fitted)")
    ax4.set_xlabel("lag time [s]")
    ax4.set_ylabel("C_φ(τ)")
    ax4.set_title("Orientation ACF  ⟨cos(Δφ(τ))⟩ — hPSC active particle")
    ax4.legend(fontsize=9)
    ax4.grid(True, ls="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig(_HERE / "correlations.png", dpi=150)
    plt.close(fig4)
    print("Saved correlations.png")

    # ── 5. Stop vs slip: x-position density reveals the boundary difference ───
    # Trajectory plots look identical for both modes — both keep the particle
    # inside the box. The physical difference is WHERE the particle accumulates:
    #   stop → clips x_new to ±L (density peak AT the wall)
    #   slip → keeps x at previous position (peak one step INSIDE the wall)
    # To make this visible, we use extreme parameters so the step v·dt is a
    # large fraction of the box width.
    V_DEMO   = 0.12   # µm/s — 30× main v;  step = v·dt = 1.2 µm
    BOX_DEMO = 1.5    # µm half-width  →  step/box = 1.2/3 = 40 %
    N_DEMO   = 400    # 400 × 10 s = 4000 s  (> 2 τ_R, enough for equilibration)
    ENS_DEMO = 200

    demo_fig, demo_axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, mode in enumerate(["stop", "slip"]):
        traj_ax = demo_axes[0, col]
        hist_ax = demo_axes[1, col]
        L = BOX_DEMO
        color = "tab:blue" if mode == "stop" else "tab:orange"

        traj = _run(ActiveBrownianParticle,
                    SimParams(D_T=D_T, D_R=D_R, v=V_DEMO, dt=DT_SIM, n_steps=N_DEMO,
                              seed=42, boundary=mode, box_size=L))
        traj_ax.plot(traj[:, 0], traj[:, 1], lw=0.4, color=color, alpha=0.8)
        rect = mpatches.FancyBboxPatch(
            (-L, -L), 2 * L, 2 * L,
            boxstyle="square,pad=0", fill=False,
            edgecolor="black", linewidth=1.5, linestyle="--",
        )
        traj_ax.add_patch(rect)
        traj_ax.set_xlim(-L * 1.2, L * 1.2)
        traj_ax.set_ylim(-L * 1.2, L * 1.2)
        traj_ax.set_aspect("equal")
        traj_ax.set_title(f"boundary='{mode}' (single trajectory)")
        traj_ax.set_xlabel("x [µm]")
        traj_ax.set_ylabel("y [µm]")

        # ensemble x-positions (all time steps pooled)
        all_x = np.concatenate([
            _run(ActiveBrownianParticle,
                 SimParams(D_T=D_T, D_R=D_R, v=V_DEMO, dt=DT_SIM, n_steps=N_DEMO,
                           seed=s, boundary=mode, box_size=L))[:, 0]
            for s in range(ENS_DEMO)
        ])
        hist_ax.hist(all_x, bins=80, range=(-L, L), density=True, alpha=0.7, color=color,
                     label=mode)
        hist_ax.axvline(-L, color="black", ls="--", lw=1.5, label="wall (±L)")
        hist_ax.axvline( L, color="black", ls="--", lw=1.5)
        # mark expected peak position for slip: L - v·dt
        if mode == "slip":
            peak = L - V_DEMO * DT_SIM
            hist_ax.axvline(peak, color="red", ls=":", lw=1.5,
                            label=f"L − v·dt = {peak:.2f} µm")
            hist_ax.axvline(-peak, color="red", ls=":", lw=1.5)
        hist_ax.set_xlabel("x [µm]")
        hist_ax.set_ylabel("probability density [µm⁻¹]")
        hist_ax.set_title(f"boundary='{mode}' (x-density,  N={ENS_DEMO} ensemble)")
        hist_ax.legend(fontsize=8)

    step_pct = V_DEMO * DT_SIM / (2 * BOX_DEMO) * 100
    demo_fig.suptitle(
        f"Stop vs Slip — x-position density shows the difference\n"
        f"v = {V_DEMO} µm/s,  box half-width L = {BOX_DEMO} µm  "
        f"→  step v·dt = {V_DEMO*DT_SIM:.2f} µm  ({step_pct:.0f}% of box width)"
    )
    demo_fig.tight_layout()
    demo_fig.savefig(_HERE / "stop_vs_slip_demo.png", dpi=150)
    plt.close(demo_fig)
    print("Saved stop_vs_slip_demo.png")

    # ── 6. MSD theory comparison: Stokes-Einstein D_R vs Sde Boker biological D_R ──
    # Stokes-Einstein: D_R and V from the global parameter block above (USE_PHYSICAL=True)
    # Sde Boker:  τ_R = 90 min from PDF memory window (page 37), v = 30 µm/h from velocity histogram
    D_R_BIO   = 1.0 / 5400.0   # rad²/s  (τ_R = 90 min)
    V_BIO     = 0.0083          # µm/s    (= 30 µm/h)
    tau_R_SE  = rotational_relaxation_time(D_R)
    tau_R_bio = rotational_relaxation_time(D_R_BIO)
    D_eff_SE  = effective_diffusion(D_T, D_R,     V)
    D_eff_bio = effective_diffusion(D_T, D_R_BIO, V_BIO)

    t_cmp = np.logspace(np.log10(DT_SIM), np.log10(5 * tau_R_bio), 800)

    msd_passive_cmp  = passive_msd(t_cmp, D_T)
    msd_active_SE    = active_msd(t_cmp, D_T, D_R,     V)
    msd_active_bio   = active_msd(t_cmp, D_T, D_R_BIO, V_BIO)
    alpha_SE_cmp     = active_msd_exponent(t_cmp, D_T, D_R,     V)
    alpha_bio_cmp    = active_msd_exponent(t_cmp, D_T, D_R_BIO, V_BIO)

    fig6, (ax_m, ax_a) = plt.subplots(2, 1, figsize=(9, 10), sharex=False)

    ax_m.loglog(t_cmp, msd_passive_cmp, color="tab:blue", lw=1.5, ls="--",
                label="Passive  (4·D_T·t,  same for both)")
    ax_m.loglog(t_cmp, msd_active_SE, color="darkorange", lw=2,
                label=f"Active — Stokes-Einstein  D_R={D_R:.2e} rad²/s, τ_R={tau_R_SE:.0f} s ({tau_R_SE/60:.0f} min)")
    ax_m.loglog(t_cmp, msd_active_bio, color="tab:red", lw=2,
                label=f"Active — Sde Boker  D_R={D_R_BIO:.2e} rad²/s, τ_R={tau_R_bio:.0f} s ({tau_R_bio/60:.0f} min)")
    ax_m.loglog(t_cmp, active_msd_long_time(t_cmp, D_T, D_R,     V),
                color="darkorange", ls="-.", lw=1.2, alpha=0.6,
                label=f"SE long-time 4·D_eff·t  (D_eff/D_T={D_eff_SE/D_T:.2f})")
    ax_m.loglog(t_cmp, active_msd_long_time(t_cmp, D_T, D_R_BIO, V_BIO),
                color="tab:red", ls="-.", lw=1.2, alpha=0.6,
                label=f"Bio long-time 4·D_eff·t  (D_eff/D_T={D_eff_bio/D_T:.2f})")
    ax_m.axvline(tau_R_SE,  color="darkorange", ls="--", lw=1, alpha=0.5,
                 label=f"τ_R SE  = {tau_R_SE:.0f} s")
    ax_m.axvline(tau_R_bio, color="tab:red",    ls="--", lw=1, alpha=0.5,
                 label=f"τ_R Bio = {tau_R_bio:.0f} s")
    ax_m.set_ylabel("MSD [µm²]")
    ax_m.set_title("MSD theory: Stokes-Einstein D_R  vs  Sde Boker biological D_R\n"
                   f"(pure theory — no simulation;  D_T={D_T:.4f} µm²/s from Stokes-Einstein throughout)")
    ax_m.legend(fontsize=7)
    ax_m.grid(True, which="both", ls="--", alpha=0.4)

    ax_a.semilogx(t_cmp, np.ones_like(t_cmp), color="tab:blue", lw=1.5, ls="--",
                  label="Passive  (α = 1)")
    ax_a.semilogx(t_cmp, alpha_SE_cmp,  color="darkorange", lw=2,
                  label=f"Active SE   (peak α = {alpha_SE_cmp.max():.3f})")
    ax_a.semilogx(t_cmp, alpha_bio_cmp, color="tab:red",    lw=2,
                  label=f"Active Bio  (peak α = {alpha_bio_cmp.max():.3f})")
    ax_a.axvline(tau_R_SE,  color="darkorange", ls="--", lw=1, alpha=0.5, label=f"τ_R SE  = {tau_R_SE:.0f} s")
    ax_a.axvline(tau_R_bio, color="tab:red",    ls="--", lw=1, alpha=0.5, label=f"τ_R Bio = {tau_R_bio:.0f} s")
    ax_a.axhline(1, color="gray", ls=":",  lw=1, alpha=0.6, label="α = 1  (diffusive)")
    ax_a.axhline(2, color="gray", ls="--", lw=1, alpha=0.6, label="α = 2  (ballistic)")
    ax_a.set_xlabel("time [s]")
    ax_a.set_ylabel("local exponent  α(t)")
    ax_a.set_title("Local MSD exponent  α(t) = d(log MSD) / d(log t)")
    ax_a.set_ylim(0.8, 2.2)
    ax_a.legend(fontsize=8)
    ax_a.grid(True, which="both", ls="--", alpha=0.4)

    fig6.tight_layout()
    fig6.savefig(_HERE / "msd_comparison_theory.png", dpi=150)
    plt.close(fig6)
    print("Saved msd_comparison_theory.png")


if __name__ == "__main__":
    main()
