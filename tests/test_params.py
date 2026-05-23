import math
import pytest
from simulation.params import SimParams


def test_boundary_default_is_none():
    p = SimParams()
    assert p.boundary == "none"
    assert p.box_size is None


def test_boundary_invalid_raises():
    with pytest.raises(ValueError, match="boundary must be one of"):
        SimParams(boundary="bounce")


def test_boundary_without_box_size_raises():
    with pytest.raises(ValueError, match="box_size must be set"):
        SimParams(boundary="reflect")


def test_boundary_with_box_size_ok():
    p = SimParams(boundary="reflect", box_size=10.0)
    assert p.boundary == "reflect"
    assert p.box_size == 10.0


def test_all_valid_boundary_values():
    for mode in ("none", "reflect", "stop", "slip"):
        box = 5.0 if mode != "none" else None
        p = SimParams(boundary=mode, box_size=box)
        assert p.boundary == mode


def test_from_physical_reference_values():
    """R=1 µm, water at 300 K: D_T ≈ 0.214 µm²/s, D_R ≈ 0.161 rad²/s.

    The stated reference values are rounded; the actual Stokes-Einstein-Debye
    result is D_T ≈ 0.2197 µm²/s and D_R ≈ 0.1648 rad²/s, so we allow ±0.01.
    """
    p = SimParams.from_physical(radius_um=1.0, T_K=300.0, eta_Pa_s=1e-3)
    assert abs(p.D_T - 0.214) < 0.01
    assert abs(p.D_R - 0.161) < 0.01


def test_from_physical_stokes_einstein_ratio():
    """D_T [m²/s] / D_R [rad²/s] must equal 4R²/3 exactly (Stokes-Einstein-Debye)."""
    radius_um = 2.0
    p = SimParams.from_physical(radius_um=radius_um)
    R_m = radius_um * 1e-6
    expected_ratio = 4 * R_m**2 / 3
    actual_ratio = (p.D_T * 1e-12) / p.D_R   # convert D_T from µm²/s to m²/s
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10


def test_from_physical_passes_kwargs():
    p = SimParams.from_physical(radius_um=1.0, v=2.0, seed=7, n_steps=500)
    assert p.v == 2.0
    assert p.seed == 7
    assert p.n_steps == 500


def test_from_physical_existing_fields_unchanged():
    """Fields not in the Stokes-Einstein formulas must keep their defaults."""
    p = SimParams.from_physical(radius_um=1.0)
    assert p.dt == 0.01
    assert p.x0 == 0.0
    assert p.boundary == "none"
