import numpy as np


def orientation_acf(
    trajs: list[np.ndarray],
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation autocorrelation: <cos(phi(t+tau) - phi(t))>.

    Args:
        trajs:   list of trajectory arrays, each shape (N+1, 3), columns [x, y, phi].
        max_lag: number of lag steps (default: N//2).

    Returns:
        (lag_steps, mean_acf) — lag_steps is integer indices; multiply by dt for seconds.
    """
    N = trajs[0].shape[0] - 1
    if max_lag is None:
        max_lag = N // 2

    acfs = []
    for traj in trajs:
        cos_phi = np.cos(traj[:, 2])
        sin_phi = np.sin(traj[:, 2])
        acf = np.empty(max_lag)
        for lag in range(max_lag):
            n = N + 1 - lag
            acf[lag] = np.mean(
                cos_phi[:n] * cos_phi[lag : lag + n]
                + sin_phi[:n] * sin_phi[lag : lag + n]
            )
        acfs.append(acf)

    return np.arange(max_lag), np.mean(acfs, axis=0)


def velocity_acf(
    trajs: list[np.ndarray],
    dt: float,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity autocorrelation from finite-difference velocities: <v(t+tau) · v(t)>.

    v(t) = (r(t+dt) - r(t)) / dt captures both thermal noise and self-propulsion.

    Args:
        trajs:   list of trajectory arrays, each shape (N+1, 3).
        dt:      simulation time step [s].
        max_lag: number of lag steps (default: N//2).

    Returns:
        (lag_steps, mean_acf) with units [µm²/s²].
    """
    N = trajs[0].shape[0] - 1
    if max_lag is None:
        max_lag = N // 2

    acfs = []
    for traj in trajs:
        vx = np.diff(traj[:, 0]) / dt  # shape (N,)
        vy = np.diff(traj[:, 1]) / dt
        acf = np.empty(max_lag)
        for lag in range(max_lag):
            n = N - lag
            acf[lag] = np.mean(
                vx[:n] * vx[lag : lag + n] + vy[:n] * vy[lag : lag + n]
            )
        acfs.append(acf)

    return np.arange(max_lag), np.mean(acfs, axis=0)


def position_acf(
    trajs: list[np.ndarray],
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Position autocorrelation: <r(t) · r(t+tau)>.

    No analytical closed form for free diffusion; grows with time for passive particles.
    Useful for qualitative comparison between boundary and particle modes.

    Args:
        trajs:   list of trajectory arrays, each shape (N+1, 3).
        max_lag: number of lag steps (default: N//2).

    Returns:
        (lag_steps, mean_acf) with units [µm²].
    """
    N = trajs[0].shape[0] - 1
    if max_lag is None:
        max_lag = N // 2

    acfs = []
    for traj in trajs:
        x, y = traj[:, 0], traj[:, 1]
        acf = np.empty(max_lag)
        for lag in range(max_lag):
            n = N + 1 - lag
            acf[lag] = np.mean(
                x[:n] * x[lag : lag + n] + y[:n] * y[lag : lag + n]
            )
        acfs.append(acf)

    return np.arange(max_lag), np.mean(acfs, axis=0)
