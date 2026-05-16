import numpy as np
import matplotlib.pyplot as plt

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from visualization.plotter import plot_trajectory, plot_msd

D_T = 0.22   # translational diffusion coefficient [µm²/s]
D_R = 0.16   # rotational diffusion coefficient [rad²/s]
V   = 2.0    # self-propulsion speed [µm/s]  — try 0, 1, 2, 3
DT_SIM = 0.01
N_STEPS = 2000
N_ENSEMBLE = 50


def run(particle_cls, params: SimParams) -> np.ndarray:
    return Simulator(particle_cls(params), params).run()


def main():
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

    # --- Ensemble MSD comparison ---
    passive_trajs = [
        run(PassiveBrownianParticle, SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=s))
        for s in range(N_ENSEMBLE)
    ]
    active_trajs = [
        run(ActiveBrownianParticle, SimParams(D_T=D_T, D_R=D_R, v=V, dt=DT_SIM, n_steps=N_STEPS, seed=s))
        for s in range(N_ENSEMBLE)
    ]

    fig2, ax = plt.subplots(figsize=(7, 5))
    plot_msd(passive_trajs, DT_SIM, label="Passive (Eq. 3)",              ax=ax)
    plot_msd(active_trajs,  DT_SIM, label=f"Active v={V} µm/s (Eq. 4)",  ax=ax)
    fig2.tight_layout()
    fig2.savefig("msd_comparison.png", dpi=150)
    print("Saved msd_comparison.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
