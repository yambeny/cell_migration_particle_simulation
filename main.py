import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf
from simulation.theory import (
    passive_msd, active_msd, active_msd_short_time, active_msd_long_time,
    effective_diffusion, rotational_relaxation_time,
    orientation_acf_theory,
)
from visualization.plotter import plot_trajectory, plot_msd

D_T = 0.22
D_R = 0.16
V   = 2.0
DT_SIM      = 0.01
N_STEPS     = 2000   # for trajectory plots
N_STEPS_MSD = 5000   # longer to show ballistic → diffusive crossover
N_ENSEMBLE  = 50


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
    t_max = N_STEPS_MSD * DT_SIM
    print(f"tau_T = {tau_T:.3f} s  |  tau_R = {tau_R:.2f} s  |  D_eff = {D_eff:.3f} µm²/s  |  t_max = {t_max:.0f} s ({t_max/tau_R:.1f}× tau_R)")

    # ── 1. Single trajectory side-by-side ─────────────────────────────────────
    passive_params = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    active_params  = SimParams(D_T=D_T, D_R=D_R, v=V,   dt=DT_SIM, n_steps=N_STEPS, seed=42)

    traj_p = _run(PassiveBrownianParticle, passive_params)
    traj_a = _run(ActiveBrownianParticle,  active_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_trajectory(traj_p, title="Passive Brownian (Eq. 3)",             ax=axes[0])
    plot_trajectory(traj_a, title=f"Active Brownian v={V} µm/s (Eq. 4)", ax=axes[1])
    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=150)
    plt.close(fig)
    print("Saved trajectories.png")

    # ── 2. Ensemble MSD with ballistic and diffusive regime overlays ───────────
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

    fig2, ax = plt.subplots(figsize=(9, 6))
    plot_msd(
        passive_trajs, DT_SIM, label=f"Passive (Eq. 3, N={N_ENSEMBLE})", ax=ax,
        theory_curves=[
            (t_th, passive_msd(t_th, D_T),
             "4·D_T·t  (theory, slope 1)",
             dict(color="navy", ls="--", lw=1.5, alpha=0.8)),
        ],
    )
    plot_msd(
        active_trajs, DT_SIM, label=f"Active v={V} µm/s (Eq. 4, N={N_ENSEMBLE})", ax=ax,
        theory_curves=[
            (t_th, active_msd(t_th, D_T, D_R, V),
             "Exact theory",
             dict(color="darkred", ls="--", lw=1.5, alpha=0.8)),
            (t_th, active_msd_short_time(t_th, D_T, V),
             f"Short-time: 4D_T·t + v²t²  (slope 2, t≪τ_R)",
             dict(color="gray", ls=":", lw=1.5)),
            (t_th, active_msd_long_time(t_th, D_T, D_R, V),
             f"Long-time: 4·D_eff·t  (slope 1, t≫τ_R)",
             dict(color="gray", ls="-.", lw=1.5)),
        ],
    )
    ax.axvline(tau_T, color="steelblue", ls=":", lw=1, alpha=0.6,
               label=f"τ_T = 4D_T/v² = {tau_T:.3f} s")
    ax.axvline(tau_R, color="black", ls="--", lw=1, alpha=0.5,
               label=f"τ_R = 1/D_R = {tau_R:.2f} s")
    ax.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig("msd_comparison.png", dpi=150)
    plt.close(fig2)
    print("Saved msd_comparison.png")

    # ── 3. Boundary condition comparison — passive and active ─────────────────
    # n_steps_msd uses N_STEPS_MSD (50 s) so the reflect saturation is visible.
    # Wall-crossing timescale: tau_cross ~ L²/(4·D_T) = 25/0.88 ≈ 28 s.
    base_p = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    fig3p = compare_boundary_conditions(
        PassiveBrownianParticle, base_p, box_size=5.0, n_ensemble=20,
        n_steps_msd=N_STEPS_MSD,
    )
    fig3p.savefig("boundary_comparison_passive.png", dpi=150)
    plt.close(fig3p)
    print("Saved boundary_comparison_passive.png")

    base_a = SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    fig3a = compare_boundary_conditions(
        ActiveBrownianParticle, base_a, box_size=5.0, n_ensemble=20,
        n_steps_msd=N_STEPS_MSD,
    )
    fig3a.savefig("boundary_comparison_active.png", dpi=150)
    plt.close(fig3a)
    print("Saved boundary_comparison_active.png")

    # ── 4. Orientation ACF + correlation time extraction ──────────────────────
    max_lag = 300
    dt_arr  = np.arange(max_lag) * DT_SIM

    _, acf_o = orientation_acf(active_trajs, max_lag=max_lag)

    # Log-linear fit: log C_φ(τ) = −D_R · τ  →  slope = −D_R
    mask    = acf_o > 0.05   # exclude noisy tail where log is unreliable
    D_R_fit = -np.polyfit(dt_arr[mask], np.log(acf_o[mask]), 1)[0]
    tau_c   = 1.0 / D_R_fit
    print(f"Fitted D_R = {D_R_fit:.3f} rad^2/s  (input = {D_R})   tau_c = {tau_c:.2f} s")

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(dt_arr, acf_o, label="simulation")
    ax4.plot(dt_arr, orientation_acf_theory(dt_arr, D_R), "--",
             color="tab:orange", alpha=0.8,
             label=f"theory: exp(−D_R τ),  D_R = {D_R}")
    ax4.axvline(tau_c, ls=":", color="gray",
                label=f"fitted τ_c = {tau_c:.2f} s  (D_R = {D_R_fit:.3f})")
    ax4.set_xlabel("lag time [s]")
    ax4.set_ylabel("C_φ(τ)")
    ax4.set_title("Orientation ACF  ⟨cos(Δφ(τ))⟩")
    ax4.legend(fontsize=9)
    ax4.grid(True, ls="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig("correlations.png", dpi=150)
    plt.close(fig4)
    print("Saved correlations.png")

    # ── 5. Stop vs slip: x-position density reveals the boundary difference ───
    # Trajectory plots look identical for both modes — both keep the particle
    # inside the box. The physical difference is WHERE the particle accumulates:
    #   stop → clips x_new to ±L (density peak AT the wall)
    #   slip → keeps x at previous position (peak one step INSIDE the wall)
    # To make this visible, we use extreme parameters so the step v·dt is a
    # large fraction of the box width.
    V_DEMO   = 20.0   # µm/s  (step = v·dt = 0.2 µm)
    BOX_DEMO = 0.5    # µm half-width  →  step/box = 0.2/0.5 = 40%
    N_DEMO   = 2000
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
    demo_fig.savefig("stop_vs_slip_demo.png", dpi=150)
    plt.close(demo_fig)
    print("Saved stop_vs_slip_demo.png")


if __name__ == "__main__":
    main()
