import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_trajectory(
    traj: np.ndarray,
    title: str = "Particle Trajectory",
    color_by_time: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot 2D trajectory. Color fades blue→red over time when color_by_time=True.

    Args:
        traj: Shape (n_steps+1, 3), columns [x, y, phi].
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    x, y = traj[:, 0], traj[:, 1]

    if color_by_time:
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="coolwarm", linewidth=0.8)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)
        ax.autoscale()
        fig.colorbar(lc, ax=ax, label="time (normalized)")
    else:
        ax.plot(x, y, lw=0.8)

    ax.set_aspect("equal")
    ax.scatter([x[0]], [y[0]], color="green", zorder=5, label="start", s=40)
    ax.scatter([x[-1]], [y[-1]], color="red", zorder=5, label="end", s=40)
    ax.legend(fontsize=8)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(title)
    return fig


def plot_msd(
    trajs: list[np.ndarray],
    dt: float,
    label: str = "",
    ax: plt.Axes | None = None,
    theory_curves: list[tuple] | None = None,
) -> plt.Figure:
    """Plot ensemble-averaged MSD vs time on a log-log scale.

    Computes displacement from each trajectory's starting point and averages
    over the ensemble. Equivalent to true MSD when all particles start from
    the same initial position (the standard setup for these simulations).

    Args:
        trajs:         List of trajectory arrays, each shape (n_steps+1, 3).
        dt:            Time step in seconds.
        label:         Legend label for the simulation data.
        theory_curves: Optional list of (t, msd, label, kwargs) tuples to
                       overlay as reference lines (e.g. theoretical predictions).
                       Each tuple: t and msd are np.ndarrays, label is a str,
                       kwargs is a dict passed directly to ax.loglog.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    n_steps = trajs[0].shape[0] - 1
    t = np.arange(1, n_steps + 1) * dt

    msds = np.array([
        (traj[1:, 0] - traj[0, 0]) ** 2 + (traj[1:, 1] - traj[0, 1]) ** 2
        for traj in trajs
    ])

    ax.loglog(t, msds.mean(axis=0), lw=2, label=label or "MSD (simulation)")

    if theory_curves:
        for t_th, msd_th, th_label, kwargs in theory_curves:
            ax.loglog(t_th, msd_th, label=th_label, **kwargs)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSD [µm²]")
    ax.set_title("Mean Squared Displacement")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return fig


def plot_correlations(
    acfs: dict[str, tuple[np.ndarray, np.ndarray]],
    theory: dict[str, np.ndarray] | None = None,
    ax_array=None,
) -> plt.Figure:
    """Plot orientation, velocity, and position autocorrelation functions.

    Args:
        acfs:     dict mapping "orientation", "velocity", "position" to
                  (lag_times_s, values) tuples where lag_times_s is in seconds.
        theory:   optional dict with same keys; values are theory arrays
                  (same length as corresponding acf arrays).
        ax_array: optional array of 3 pre-existing Axes to draw into.

    Returns:
        The Figure.
    """
    if ax_array is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = ax_array
        fig = axes[0].get_figure()

    configs = [
        ("orientation", "Orientation ACF\n⟨cos(Δφ(τ))⟩",       "C_φ(τ)"),
        ("velocity",    "Velocity ACF\n⟨v(t+τ)·v(t)⟩",          "C_v(τ) [µm²/s²]"),
        ("position",    "Position ACF\n⟨r(t)·r(t+τ)⟩",          "C_r(τ) [µm²]"),
    ]

    for ax, (key, title, ylabel) in zip(axes, configs):
        if key not in acfs:
            ax.set_visible(False)
            continue
        t, values = acfs[key]
        ax.plot(t, values, label="simulation")
        if theory and key in theory:
            ax.plot(t, theory[key], "--", label="theory", alpha=0.7)
        ax.set_xlabel("lag time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    fig.tight_layout()
    return fig
