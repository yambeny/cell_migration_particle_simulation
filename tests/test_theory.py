import numpy as np
import pytest
from simulation.theory import (
    passive_msd,
    active_msd,
    active_msd_short_time,
    active_msd_long_time,
    effective_diffusion,
    rotational_relaxation_time,
)


def test_passive_msd_linear():
    t = np.array([1.0, 2.0, 4.0])
    result = passive_msd(t, D_T=0.22)
    np.testing.assert_allclose(result, 4 * 0.22 * t)


def test_active_msd_reduces_to_passive_when_v0():
    t = np.linspace(0.01, 50.0, 200)
    np.testing.assert_allclose(
        active_msd(t, D_T=0.22, D_R=0.16, v=0.0),
        passive_msd(t, D_T=0.22),
        rtol=1e-10,
    )


def test_active_msd_short_time_ballistic():
    """For t << tau_R, active_msd ≈ active_msd_short_time = 4*D_T*t + v²t²."""
    D_T, D_R, v = 0.22, 0.16, 2.0
    t = np.array([0.001, 0.01, 0.1])  # t << tau_R = 6.25 s

    # At short times exp(-D_R*t) ≈ 1 - D_R*t + ..., so exact → 4*D_T*t + v²t²
    np.testing.assert_allclose(
        active_msd(t, D_T, D_R, v),
        active_msd_short_time(t, D_T, v),
        rtol=0.01,
    )


def test_active_msd_long_time_slope():
    """For t >> tau_R, d(MSD)/dt → 4*D_eff (slope 1 with enhanced diffusivity)."""
    D_T, D_R, v = 0.22, 0.16, 2.0
    # Use two widely spaced large-t points; constant offset −2v²/D_R² becomes negligible
    t = np.array([10000.0, 20000.0])
    msd = active_msd(t, D_T, D_R, v)
    slope = (msd[1] - msd[0]) / (t[1] - t[0])
    D_eff = effective_diffusion(D_T, D_R, v)
    assert abs(slope - 4.0 * D_eff) / (4.0 * D_eff) < 1e-4


def test_active_msd_short_time_matches_asymptote():
    t = np.linspace(0.001, 0.1, 50)
    D_T, D_R, v = 0.22, 0.16, 2.0
    np.testing.assert_allclose(
        active_msd_short_time(t, D_T, v),
        4 * D_T * t + v**2 * t**2,
        rtol=1e-12,
    )


def test_effective_diffusion():
    D_eff = effective_diffusion(D_T=0.22, D_R=0.16, v=2.0)
    expected = 0.22 + 4.0 / (2 * 0.16)   # = 0.22 + 12.5 = 12.72
    assert abs(D_eff - expected) < 1e-10


def test_rotational_relaxation_time():
    assert abs(rotational_relaxation_time(0.16) - 6.25) < 1e-10


def test_orientation_acf_theory_at_zero():
    from simulation.theory import orientation_acf_theory
    t = np.array([0.0, 1.0, 2.0])
    result = orientation_acf_theory(t, D_R=0.5)
    assert abs(result[0] - 1.0) < 1e-10   # exp(0) = 1


def test_orientation_acf_theory_decays():
    from simulation.theory import orientation_acf_theory
    t = np.linspace(0, 10, 100)
    result = orientation_acf_theory(t, D_R=0.5)
    assert result[-1] < result[0]
    np.testing.assert_allclose(result, np.exp(-0.5 * t))


def test_velocity_acf_theory_at_zero():
    from simulation.theory import velocity_acf_theory
    t = np.array([0.0])
    result = velocity_acf_theory(t, v=2.0, D_R=0.5)
    assert abs(result[0] - 4.0) < 1e-10   # v² * exp(0) = 4


def test_velocity_acf_theory_decays():
    from simulation.theory import velocity_acf_theory
    t = np.linspace(0, 10, 100)
    result = velocity_acf_theory(t, v=2.0, D_R=0.5)
    np.testing.assert_allclose(result, 4.0 * np.exp(-0.5 * t))
