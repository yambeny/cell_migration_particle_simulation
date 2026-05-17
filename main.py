import numpy as np
import matplotlib.pyplot as plt

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.theory import (
    passive_msd,
    active_msd,
    active_msd_short_time,
    active_msd_long_time,
    effective_diffusion,
    rotational_relaxation_time,
)
from visualization.plotter import plot_trajectory, plot_msd

D_T = 0.22   # translational diffusion coefficient [µm²/s]
D_R = 0.16   # rotational diffusion coefficient [rad²/s]
V   = 2.0    # self-propulsion speed [µm/s]  — try 0, 1, 2, 3
DT_SIM    = 0.01
N_STEPS   = 2000   # trajectory plots
N_STEPS_MSD = 5000 # MSD plots — longer to show both ballistic and diffusive regimes
N_ENSEMBLE  = 50


def run(particle_cls, params: SimParams) -> np.ndarray:
    return Simulator(particle_cls(params), params).run()


def main():
    # --- Derived theory quantities ---
    tau_R  = rotational_relaxation_time(D_R)          # crossover time [s]
    D_eff  = effective_diffusion(D_T, D_R, V)          # enhanced diffusivity [µm²/s]
    t_max  = N_STEPS_MSD * DT_SIM
    print(f"tau_R = {tau_R:.2f} s  |  D_eff = {D_eff:.3f} um^2/s  |  t_max = {t_max:.0f} s ({t_max/tau_R:.1f}x tau_R)")

    # --- Single trajectory side-by-side ---
    passive_params = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    active_params  = SimParams(D_T=D_T, D_R=D_R, v=V,   dt=DT_SIM, n_steps=N_STEPS, seed=42)

    traj_p = run(PassiveBrownianParticle, passive_params)
    traj_a = run(ActiveBrownianParticle,  active_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_trajectory(traj_p, title="Passive Brownian (Eq. 3)",             ax=axes[0])
    plot_trajectory(traj_a, title=f"Active Brownian v={V} µm/s (Eq. 4)", ax=axes[1])
    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=150)
    print("Saved trajectories.png")
    plt.close(fig)

    # --- Ensemble MSD with theoretical predictions ---
    passive_trajs = [
        run(PassiveBrownianParticle,
            SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS_MSD, seed=s))
        for s in range(N_ENSEMBLE)
    ]
    active_trajs = [
        run(ActiveBrownianParticle,
            SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS_MSD, seed=s))
        for s in range(N_ENSEMBLE)
    ]

    t_th = np.arange(1, N_STEPS_MSD + 1) * DT_SIM   # theory time axis

    passive_theory = [
        (t_th, passive_msd(t_th, D_T),
         f"4·D_T·t  (theory, slope 1)",
         {"color": "navy", "ls": "--", "lw": 1.5, "alpha": 0.8}),
    ]
    active_theory = [
        (t_th, active_msd(t_th, D_T, D_R, V),
         "Exact theory",
         {"color": "darkred", "ls": "--", "lw": 1.5, "alpha": 0.8}),
        (t_th, active_msd_short_time(t_th, D_T, V),
         f"Short-time: 4D_T·t + v²t²  (slope 2, t≪τ_R)",
         {"color": "gray", "ls": ":", "lw": 1.5}),
        (t_th, active_msd_long_time(t_th, D_T, D_R, V),
         f"Long-time: 4·D_eff·t  (slope 1, t≫τ_R)",
         {"color": "gray", "ls": "-.", "lw": 1.5}),
    ]

    fig2, ax = plt.subplots(figsize=(9, 6))
    plot_msd(passive_trajs, DT_SIM,
             label=f"Passive (Eq. 3, sim N={N_ENSEMBLE})",
             ax=ax, theory_curves=passive_theory)
    plot_msd(active_trajs, DT_SIM,
             label=f"Active v={V} µm/s (Eq. 4, sim N={N_ENSEMBLE})",
             ax=ax, theory_curves=active_theory)

    ax.axvline(tau_R, color="black", ls="--", lw=1, alpha=0.5,
               label=f"τ_R = 1/D_R = {tau_R:.2f} s")
    ax.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig("msd_comparison.png", dpi=150)
    print("Saved msd_comparison.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
