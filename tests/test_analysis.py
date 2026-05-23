import numpy as np
import pytest
from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf, velocity_acf, position_acf


def _run(seed=0, D_T=0.22, D_R=0.16, v=0.0, n_steps=200, dt=0.01):
    params = SimParams(D_T=D_T, D_R=D_R, v=v, dt=dt, n_steps=n_steps, seed=seed)
    cls = ActiveBrownianParticle if v > 0 else PassiveBrownianParticle
    return Simulator(cls(params), params).run()


# ── orientation_acf ──────────────────────────────────────────────────────────

def test_orientation_acf_shape():
    traj = _run()
    lag_steps, acf = orientation_acf([traj], max_lag=20)
    assert lag_steps.shape == (20,)
    assert acf.shape == (20,)
    assert list(lag_steps) == list(range(20))


def test_orientation_acf_zero_lag_is_one():
    traj = _run()
    _, acf = orientation_acf([traj], max_lag=10)
    assert abs(acf[0] - 1.0) < 1e-10


def test_orientation_acf_no_rotation_stays_one():
    """D_R=0: orientation never changes, ACF=1 at all lags."""
    traj = _run(D_R=0.0)
    _, acf = orientation_acf([traj], max_lag=10)
    np.testing.assert_allclose(acf, 1.0, atol=1e-10)


def test_orientation_acf_decays_with_rotation():
    """High D_R: ensemble ACF at last lag must be less than at lag 0."""
    trajs = [_run(seed=s, D_R=2.0, n_steps=300) for s in range(40)]
    _, acf = orientation_acf(trajs, max_lag=50)
    assert acf[-1] < acf[0]


# ── velocity_acf ─────────────────────────────────────────────────────────────

def test_velocity_acf_shape():
    traj = _run()
    lag_steps, acf = velocity_acf([traj], dt=0.01, max_lag=20)
    assert lag_steps.shape == (20,)
    assert acf.shape == (20,)


def test_velocity_acf_zero_lag_positive():
    """Zero-lag velocity ACF = mean(v²) > 0 for non-zero diffusion."""
    traj = _run(D_T=0.22)
    _, acf = velocity_acf([traj], dt=0.01, max_lag=5)
    assert acf[0] > 0.0


# ── position_acf ─────────────────────────────────────────────────────────────

def test_position_acf_shape():
    traj = _run()
    lag_steps, acf = position_acf([traj], max_lag=20)
    assert lag_steps.shape == (20,)
    assert acf.shape == (20,)


def test_position_acf_zero_lag_nonnegative():
    """Zero-lag position ACF = mean(r²) >= 0."""
    traj = _run(n_steps=500)
    _, acf = position_acf([traj], max_lag=10)
    assert acf[0] >= 0.0


def test_position_acf_stationary_particle_is_zero():
    """Particle that never moves: position ACF = 0 at all lags."""
    traj = _run(D_T=0.0, D_R=0.0, v=0.0)
    _, acf = position_acf([traj], max_lag=10)
    np.testing.assert_allclose(acf, 0.0, atol=1e-10)


# ── default max_lag ───────────────────────────────────────────────────────────

def test_default_max_lag_is_half_n_steps():
    traj = _run(n_steps=100)
    _, acf = orientation_acf([traj])
    assert len(acf) == 50   # N//2 = 100//2
