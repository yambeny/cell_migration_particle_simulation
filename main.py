import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf
from simulation.theory import (
    passive_msd, active_msd, effective_diffusion, rotational_relaxation_time,
    orientation_acf_theory,
)
from visualization.plotter import plot_trajectory, plot_msd

D_T = 0.22
D_R = 0.16
V   = 2.0
DT_SIM  = 0.01
N_STEPS = 2000
N_ENSEMBLE = 50


def _run(particle_cls, params: SimParams) -> np.ndarray:
    return Simulator(particle_cls(params), params).run()


def compare_boundary_conditions(
    particle_cls,
    base_params: SimParams,
    box_size: float,
    n_ensemble: int = 30,
) -> plt.Figure:
    """Run particle_cls under all four boundary modes and compare trajectories + MSD.

    Args:
        particle_cls: PassiveBrownianParticle or ActiveBrownianParticle.
        base_params:  SimParams instance whose D_T, D_R, v, dt, n_steps are reused.
                      boundary and box_size in base_params are ignored.
        box_size:     half-width of the confinement box [µm].
        n_ensemble:   number of realizations for the MSD comparison.

    Returns:
        Figure with 2 rows: top = 4 trajectory plots, bottom = MSD comparison.
    """
    modes = ["none", "reflect", "stop", "slip"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig = plt.figure(figsize=(18, 9))
    traj_axes = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
    msd_ax = fig.add_subplot(2, 1, 2)

    def _params_for(mode):
        return SimParams(
            D_T=base_params.D_T, D_R=base_params.D_R, v=base_params.v,
            dt=base_params.dt, n_steps=base_params.n_steps,
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
    for mode, color in zip(modes, colors):
        trajs = [
            _run(particle_cls, SimParams(
                D_T=base_params.D_T, D_R=base_params.D_R, v=base_params.v,
                dt=dt, n_steps=base_params.n_steps, seed=s,
                x0=base_params.x0, y0=base_params.y0, phi0=base_params.phi0,
                boundary=mode,
                box_size=box_size if mode != "none" else None,
            ))
            for s in range(n_ensemble)
        ]
        t = np.arange(1, base_params.n_steps + 1) * dt
        msds = np.array([
            (tr[1:, 0] - tr[0, 0]) ** 2 + (tr[1:, 1] - tr[0, 1]) ** 2
            for tr in trajs
        ])
        msd_ax.loglog(t, msds.mean(axis=0), label=f"'{mode}'", color=color)

    msd_ax.set_xlabel("time [s]")
    msd_ax.set_ylabel("MSD [µm²]")
    msd_ax.set_title("MSD — all boundary conditions")
    msd_ax.legend()
    msd_ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    return fig


def main():
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
    print("Saved trajectories.png")

    # ── 2. Ensemble MSD comparison ─────────────────────────────────────────────
    tau_R  = rotational_relaxation_time(D_R)
    D_eff  = effective_diffusion(D_T, D_R, V)
    print(f"tau_R = {tau_R:.2f} s    D_eff = {D_eff:.2f} µm²/s")

    passive_trajs = [
        _run(PassiveBrownianParticle,
             SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=s))
        for s in range(N_ENSEMBLE)
    ]
    active_trajs = [
        _run(ActiveBrownianParticle,
             SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS, seed=s))
        for s in range(N_ENSEMBLE)
    ]

    t_th = np.arange(1, N_STEPS + 1) * DT_SIM
    fig2, ax = plt.subplots(figsize=(7, 5))
    plot_msd(
        passive_trajs, DT_SIM, label="Passive (Eq. 3)", ax=ax,
        theory_curves=[
            (t_th, passive_msd(t_th, D_T), "passive theory",
             dict(ls="--", color="tab:blue", alpha=0.7)),
        ],
    )
    plot_msd(
        active_trajs, DT_SIM, label=f"Active v={V} µm/s (Eq. 4)", ax=ax,
        theory_curves=[
            (t_th, active_msd(t_th, D_T, D_R, V), "active theory",
             dict(ls="--", color="tab:orange", alpha=0.7)),
        ],
    )
    fig2.tight_layout()
    fig2.savefig("msd_comparison.png", dpi=150)
    print("Saved msd_comparison.png")

    # ── 3. Boundary condition comparison ──────────────────────────────────────
    base = SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    fig3 = compare_boundary_conditions(
        ActiveBrownianParticle, base, box_size=5.0, n_ensemble=20
    )
    fig3.savefig("boundary_comparison.png", dpi=150)
    print("Saved boundary_comparison.png")

    # ── 4. Orientation ACF + correlation time extraction ──────────────────────
    max_lag = 300
    dt_arr  = np.arange(max_lag) * DT_SIM

    _, acf_o = orientation_acf(active_trajs, max_lag=max_lag)

    # Log-linear fit: log C_φ(τ) = −D_R · τ  →  slope = −D_R
    mask    = acf_o > 0.05   # exclude noisy tail where log is unreliable
    D_R_fit = -np.polyfit(dt_arr[mask], np.log(acf_o[mask]), 1)[0]
    tau_c   = 1.0 / D_R_fit
    print(f"Fitted D_R = {D_R_fit:.3f} rad²/s  (input = {D_R})   τ_c = {tau_c:.2f} s")

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
    print("Saved correlations.png")

    plt.show()


if __name__ == "__main__":
    main()
