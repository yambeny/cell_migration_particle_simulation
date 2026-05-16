import numpy as np
import pytest
from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator


def test_simulator_returns_correct_shape():
    params = SimParams(n_steps=50, seed=0)
    traj = Simulator(PassiveBrownianParticle(params), params).run()
    assert traj.shape == (51, 3)  # n_steps+1 rows (includes t=0); 3 cols: x, y, phi


def test_simulator_first_row_is_initial_state():
    params = SimParams(x0=3.0, y0=-1.0, phi0=1.5, n_steps=10, seed=0)
    traj = Simulator(PassiveBrownianParticle(params), params).run()
    assert traj[0, 0] == 3.0
    assert traj[0, 1] == -1.0
    assert traj[0, 2] == 1.5


def test_simulator_is_deterministic():
    params = SimParams(n_steps=100, v=1.0, seed=42)
    traj1 = Simulator(ActiveBrownianParticle(params), params).run()
    traj2 = Simulator(ActiveBrownianParticle(params), params).run()
    np.testing.assert_array_equal(traj1, traj2)


def test_passive_msd_scales_linearly_with_time():
    """MSD of passive particle ~4*D_T*t (ensemble average over 200 realizations)."""
    D_T = 0.22
    dt = 0.01
    n_steps = 500
    msds = []
    for seed in range(200):
        params = SimParams(D_T=D_T, D_R=0.0, v=0.0, dt=dt, n_steps=n_steps, seed=seed)
        traj = Simulator(PassiveBrownianParticle(params), params).run()
        msds.append(traj[-1, 0] ** 2 + traj[-1, 1] ** 2)
    measured = np.mean(msds)
    expected = 4 * D_T * (n_steps * dt)   # 4*D_T*t in 2D
    assert abs(measured - expected) / expected < 0.20   # 20% statistical tolerance
