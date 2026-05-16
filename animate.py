"""Live animation of passive (Eq. 3) and active (Eq. 4) Brownian particles.

Run:  python animate.py

Tweak SKIP / INTERVAL to control playback speed.
Tweak V, N_STEPS, TRAIL to explore different regimes.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator

# --- Parameters (edit freely) ---
D_T      = 0.22    # translational diffusion coefficient [µm²/s]
D_R      = 0.16    # rotational diffusion coefficient [rad²/s]
V        = 2.0     # self-propulsion speed [µm/s]  — set to 0 for pure diffusion
DT_SIM   = 0.01    # time step [s]
N_STEPS  = 3000    # total simulation steps
TRAIL    = 200     # number of past positions kept in the visible trail
SKIP     = 3       # simulation steps per animation frame (higher = faster playback)
INTERVAL = 30      # milliseconds between frames


def main():
    # Pre-compute full trajectories before animating
    p_params = SimParams(D_T=D_T, D_R=D_R, v=0.0, dt=DT_SIM, n_steps=N_STEPS, seed=42)
    a_params = SimParams(D_T=D_T, D_R=D_R, v=V,   dt=DT_SIM, n_steps=N_STEPS, seed=42)
    traj_p = Simulator(PassiveBrownianParticle(p_params), p_params).run()
    traj_a = Simulator(ActiveBrownianParticle(a_params),  a_params).run()

    # Fixed axis limits from both trajectories so the view doesn't jump
    all_x = np.concatenate([traj_p[:, 0], traj_a[:, 0]])
    all_y = np.concatenate([traj_p[:, 1], traj_a[:, 1]])
    pad = 0.12 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 1.0)
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)
    arrow_len = 0.08 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 1.0)

    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title in [
        (ax_p, "Passive Brownian (Eq. 3)"),
        (ax_a, f"Active Brownian  v = {V} µm/s  (Eq. 4)"),
    ]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.grid(True, ls="--", alpha=0.3)

    # Passive artists
    trail_p,  = ax_p.plot([], [], lw=0.8, color="steelblue", alpha=0.6)
    dot_p,    = ax_p.plot([], [], "o",  color="steelblue",   ms=7, zorder=5)
    time_lbl   = ax_p.text(0.03, 0.95, "", transform=ax_p.transAxes,
                            fontsize=9, va="top")

    # Active artists
    trail_a,  = ax_a.plot([], [], lw=0.8, color="tomato", alpha=0.6)
    dot_a,    = ax_a.plot([], [], "o",  color="tomato",   ms=7, zorder=5)
    orient_a, = ax_a.plot([], [], "-",  color="darkred",  lw=2,  zorder=6)

    def init():
        trail_p.set_data([], [])
        dot_p.set_data([], [])
        trail_a.set_data([], [])
        dot_a.set_data([], [])
        orient_a.set_data([], [])
        time_lbl.set_text("")
        return trail_p, dot_p, trail_a, dot_a, orient_a, time_lbl

    def update(frame):
        i = min(frame * SKIP, N_STEPS)
        s = max(0, i - TRAIL)

        trail_p.set_data(traj_p[s:i+1, 0], traj_p[s:i+1, 1])
        dot_p.set_data([traj_p[i, 0]], [traj_p[i, 1]])

        trail_a.set_data(traj_a[s:i+1, 0], traj_a[s:i+1, 1])
        dot_a.set_data([traj_a[i, 0]], [traj_a[i, 1]])

        # Orientation arrow showing current phi
        phi = traj_a[i, 2]
        x, y = traj_a[i, 0], traj_a[i, 1]
        orient_a.set_data(
            [x, x + arrow_len * np.cos(phi)],
            [y, y + arrow_len * np.sin(phi)],
        )

        time_lbl.set_text(f"t = {i * DT_SIM:.1f} s")
        return trail_p, dot_p, trail_a, dot_a, orient_a, time_lbl

    n_frames = N_STEPS // SKIP + 1
    # Keep reference to ani so it isn't garbage-collected before plt.show()
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=INTERVAL, blit=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
