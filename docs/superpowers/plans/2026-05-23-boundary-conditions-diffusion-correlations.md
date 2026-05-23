# Boundary Conditions, Physical Diffusion Mode & Correlation Functions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three boundary condition modes (reflect/stop/slip), a physical D_T/D_R classmethod, three correlation functions with theory overlays, and a boundary comparison helper to the existing particle simulation.

**Architecture:** `SimParams` grows two new fields and a classmethod; `Particle` base class gets `_apply_boundary()` called from both `step()` implementations; a new `simulation/analysis.py` holds correlation function computation; `simulation/theory.py` and `visualization/plotter.py` each gain two new functions; `main.py` gains `compare_boundary_conditions()`.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, pytest

---

## File Map

| File | Change |
|---|---|
| `simulation/params.py` | Add `boundary`, `box_size` fields; `__post_init__` validation; `from_physical` classmethod |
| `simulation/particle.py` | Add `_apply_boundary()` to base; refactor `step()` in both subclasses to call it |
| `simulation/analysis.py` | **New.** `orientation_acf`, `velocity_acf`, `position_acf` |
| `simulation/theory.py` | Add `orientation_acf_theory`, `velocity_acf_theory` |
| `visualization/plotter.py` | Add `plot_correlations` |
| `main.py` | Add `compare_boundary_conditions`; add demo calls |
| `tests/test_params.py` | **New.** Tests for `from_physical` and boundary validation |
| `tests/test_particle.py` | Add boundary condition tests |
| `tests/test_analysis.py` | **New.** Tests for all three ACF functions |
| `README.md` | Update file table, parameters table, tests table |

---

## Task 1: SimParams — boundary fields, validation, from_physical

**Files:**
- Modify: `simulation/params.py`
- Create: `tests/test_params.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_params.py`:
```python
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
    """R=1 µm, water at 300 K: D_T ≈ 0.214 µm²/s, D_R ≈ 0.161 rad²/s."""
    p = SimParams.from_physical(radius_um=1.0, T_K=300.0, eta_Pa_s=1e-3)
    assert abs(p.D_T - 0.214) < 0.001
    assert abs(p.D_R - 0.161) < 0.001


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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_params.py -v
```
Expected: all FAIL with `TypeError` or `AttributeError` (fields/method don't exist yet).

- [ ] **Step 3: Implement SimParams changes**

Replace the entire contents of `simulation/params.py`:
```python
from __future__ import annotations
import math
from dataclasses import dataclass

_KB = 1.380649e-23  # Boltzmann constant [J/K]
_VALID_BOUNDARIES = {"none", "reflect", "stop", "slip"}


@dataclass
class SimParams:
    D_T: float = 0.22       # translational diffusion coefficient [µm²/s]
    D_R: float = 0.16       # rotational diffusion coefficient [rad²/s]
    v: float = 0.0          # self-propulsion speed [µm/s]; 0 = passive
    dt: float = 0.01        # time step [s]
    n_steps: int = 1000     # number of simulation steps
    seed: int = 42          # RNG seed; set to None for non-reproducible runs
    x0: float = 0.0         # initial x position [µm]
    y0: float = 0.0         # initial y position [µm]
    phi0: float = 0.0       # initial orientation angle [rad]
    boundary: str = "none"  # one of: "none", "reflect", "stop", "slip"
    box_size: float | None = None  # half-width L [µm]; box = [−L, L] × [−L, L]

    def __post_init__(self) -> None:
        if self.boundary not in _VALID_BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {_VALID_BOUNDARIES}, got {self.boundary!r}"
            )
        if self.boundary != "none" and self.box_size is None:
            raise ValueError("box_size must be set when boundary != 'none'")

    @classmethod
    def from_physical(
        cls,
        radius_um: float,
        T_K: float = 300.0,
        eta_Pa_s: float = 1e-3,
        **kwargs,
    ) -> SimParams:
        """Build SimParams with D_T and D_R derived from Stokes-Einstein-Debye.

        D_T = k_B T / (6π η R)   [µm²/s]   — translational (Eq. 1)
        D_R = k_B T / (8π η R³)  [rad²/s]  — rotational    (Eq. 2)

        Args:
            radius_um:  particle radius [µm]
            T_K:        temperature [K] (default 300 K)
            eta_Pa_s:   dynamic viscosity [Pa·s] (default 1e-3, water at ~20 °C)
            **kwargs:   any other SimParams fields (v, dt, n_steps, seed, …)
        """
        R = radius_um * 1e-6  # convert µm → m
        D_T = _KB * T_K / (6.0 * math.pi * eta_Pa_s * R) * 1e12  # m²/s → µm²/s
        D_R = _KB * T_K / (8.0 * math.pi * eta_Pa_s * R**3)      # rad²/s
        return cls(D_T=D_T, D_R=D_R, **kwargs)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_params.py -v
```
Expected: all PASS.

- [ ] **Step 5: Confirm existing tests still pass**

```
python -m pytest tests/ -v
```
Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```
git add simulation/params.py tests/test_params.py
git commit -m "feat: add boundary fields, validation, and from_physical to SimParams"
```

---

## Task 2: Particle — _apply_boundary + refactored step()

**Files:**
- Modify: `simulation/particle.py`
- Modify: `tests/test_particle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_particle.py`:
```python
# ── Boundary condition tests ──────────────────────────────────────────────────

def _make(boundary, box_size=5.0, x0=0.0, y0=0.0, phi0=0.0,
           D_T=0.0, D_R=0.0, v=0.0, n_steps=1, seed=0):
    params = SimParams(
        D_T=D_T, D_R=D_R, v=v, dt=1.0, n_steps=n_steps,
        boundary=boundary, box_size=box_size,
        x0=x0, y0=y0, phi0=phi0, seed=seed,
    )
    return PassiveBrownianParticle(params)


# reflect
def test_reflect_position_mirrors_across_wall():
    """x=4.9, proposed x_new=5.1 → reflected to 4.9."""
    p = _make("reflect")
    x, y, phi = p._apply_boundary(5.1, 0.0, 0.0)
    assert abs(x - 4.9) < 1e-10
    assert y == 0.0


def test_reflect_orientation_flips_on_x_wall():
    """Hitting x-wall: phi → pi - phi."""
    p = _make("reflect")
    _, _, phi_out = p._apply_boundary(5.5, 0.0, 0.3)
    assert abs(phi_out - (np.pi - 0.3)) < 1e-10


def test_reflect_orientation_flips_on_y_wall():
    """Hitting y-wall: phi → -phi."""
    p = _make("reflect")
    _, _, phi_out = p._apply_boundary(0.0, 5.5, 0.3)
    assert abs(phi_out - (-0.3)) < 1e-10


def test_reflect_stays_inside_box():
    """Long run with high diffusion must never leave the box."""
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="reflect", box_size=5.0, seed=1)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


# stop
def test_stop_clamps_to_wall():
    """Proposed x_new=7.0 with L=5 → x clamped to 5.0."""
    p = _make("stop")
    x, y, phi = p._apply_boundary(7.0, 0.0, 0.5)
    assert x == 5.0
    assert y == 0.0
    assert phi == 0.5  # phi unchanged


def test_stop_stays_inside_box():
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="stop", box_size=5.0, seed=2)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


# slip
def test_slip_blocks_crossing_axis_keeps_parallel():
    """At x0=4.9, x_new=5.4 (exits), y_new=0.5 (stays) → x blocked, y moves."""
    p = _make("slip", x0=4.9)
    x, y, phi = p._apply_boundary(5.4, 0.5, 0.3)
    assert abs(x - 4.9) < 1e-10   # x blocked: stays at x0
    assert abs(y - 0.5) < 1e-10   # y passes through
    assert phi == 0.3              # phi unchanged


def test_slip_stays_inside_box():
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="slip", box_size=5.0, seed=3)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


def test_boundary_none_is_unchanged():
    """boundary='none' must produce bit-identical results to old behavior."""
    params_old = SimParams(D_T=0.22, D_R=0.16, v=1.0, dt=0.01, n_steps=100, seed=99)
    params_new = SimParams(D_T=0.22, D_R=0.16, v=1.0, dt=0.01, n_steps=100, seed=99,
                           boundary="none")
    from simulation.particle import ActiveBrownianParticle
    from simulation.simulator import Simulator
    traj_old = Simulator(ActiveBrownianParticle(params_old), params_old).run()
    traj_new = Simulator(ActiveBrownianParticle(params_new), params_new).run()
    np.testing.assert_array_equal(traj_old, traj_new)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_particle.py -k "boundary or reflect or stop or slip" -v
```
Expected: FAIL with `AttributeError: '_apply_boundary' not found`.

- [ ] **Step 3: Implement _apply_boundary and refactor step() methods**

Replace the entire contents of `simulation/particle.py`:
```python
from abc import ABC, abstractmethod
import numpy as np
from simulation.params import SimParams


class Particle(ABC):
    def __init__(self, params: SimParams):
        self.params = params
        self.x = params.x0
        self.y = params.y0
        self.phi = params.phi0
        self._rng = np.random.default_rng(params.seed)

    @abstractmethod
    def step(self) -> None:
        """Advance particle state by one time step dt."""

    def state(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.phi)

    def _apply_boundary(
        self, x_new: float, y_new: float, phi: float
    ) -> tuple[float, float, float]:
        """Apply the boundary condition from params and return corrected (x, y, phi)."""
        p = self.params
        if p.boundary == "none" or p.box_size is None:
            return x_new, y_new, phi

        L = p.box_size

        if p.boundary == "reflect":
            x_hit = False
            y_hit = False
            if x_new > L:
                x_new = 2.0 * L - x_new
                x_hit = True
            elif x_new < -L:
                x_new = -2.0 * L - x_new
                x_hit = True
            if y_new > L:
                y_new = 2.0 * L - y_new
                y_hit = True
            elif y_new < -L:
                y_new = -2.0 * L - y_new
                y_hit = True
            if x_hit:
                phi = np.pi - phi
            if y_hit:
                phi = -phi
            return x_new, y_new, phi

        if p.boundary == "stop":
            return float(np.clip(x_new, -L, L)), float(np.clip(y_new, -L, L)), phi

        if p.boundary == "slip":
            if x_new > L or x_new < -L:
                x_new = self.x   # block x: stay at current x
            if y_new > L or y_new < -L:
                y_new = self.y   # block y: stay at current y
            return x_new, y_new, phi

        return x_new, y_new, phi


class PassiveBrownianParticle(Particle):
    """Passive Brownian motion — Eq. 3.

    Euler-Maruyama:
        x += sqrt(2*D_T*dt) * eta_x
        y += sqrt(2*D_T*dt) * eta_y
        phi += sqrt(2*D_R*dt) * eta_phi
    """

    def step(self) -> None:
        p = self.params
        eta = self._rng.standard_normal(3)
        noise_t = np.sqrt(2.0 * p.D_T * p.dt)
        noise_r = np.sqrt(2.0 * p.D_R * p.dt)
        phi_new = self.phi + noise_r * eta[2]
        x_new = self.x + noise_t * eta[0]
        y_new = self.y + noise_t * eta[1]
        self.x, self.y, self.phi = self._apply_boundary(x_new, y_new, phi_new)


class ActiveBrownianParticle(Particle):
    """Self-propelled active Brownian particle — Eq. 4.

    Euler-Maruyama:
        x += v*cos(phi)*dt + sqrt(2*D_T*dt) * eta_x
        y += v*sin(phi)*dt + sqrt(2*D_T*dt) * eta_y
        phi += sqrt(2*D_R*dt) * eta_phi
    """

    def step(self) -> None:
        p = self.params
        eta = self._rng.standard_normal(3)
        noise_t = np.sqrt(2.0 * p.D_T * p.dt)
        noise_r = np.sqrt(2.0 * p.D_R * p.dt)
        phi_new = self.phi + noise_r * eta[2]
        x_new = self.x + p.v * np.cos(self.phi) * p.dt + noise_t * eta[0]
        y_new = self.y + p.v * np.sin(self.phi) * p.dt + noise_t * eta[1]
        self.x, self.y, self.phi = self._apply_boundary(x_new, y_new, phi_new)
```

- [ ] **Step 4: Run all particle tests**

```
python -m pytest tests/test_particle.py -v
```
Expected: all PASS (including the new boundary tests and all pre-existing tests).

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```
git add simulation/particle.py tests/test_particle.py
git commit -m "feat: add _apply_boundary to Particle; implement reflect, stop, slip modes"
```

---

## Task 3: Correlation functions (simulation/analysis.py)

**Files:**
- Create: `simulation/analysis.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis.py`:
```python
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
    """Zero-lag position ACF = mean(r²) ≥ 0."""
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_analysis.py -v
```
Expected: all FAIL with `ModuleNotFoundError: No module named 'simulation.analysis'`.

- [ ] **Step 3: Implement simulation/analysis.py**

Create `simulation/analysis.py`:
```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_analysis.py -v
```
Expected: all PASS.

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```
git add simulation/analysis.py tests/test_analysis.py
git commit -m "feat: add orientation, velocity, and position autocorrelation functions"
```

---

## Task 4: Theory — ACF analytical predictions

**Files:**
- Modify: `simulation/theory.py`

- [ ] **Step 1: Write failing tests**

Append to an existing test file. Open `tests/test_theory.py` and append:
```python
from simulation.theory import orientation_acf_theory, velocity_acf_theory


def test_orientation_acf_theory_at_zero():
    t = np.array([0.0, 1.0, 2.0])
    result = orientation_acf_theory(t, D_R=0.5)
    assert abs(result[0] - 1.0) < 1e-10   # exp(0) = 1


def test_orientation_acf_theory_decays():
    t = np.linspace(0, 10, 100)
    result = orientation_acf_theory(t, D_R=0.5)
    assert result[-1] < result[0]
    np.testing.assert_allclose(result, np.exp(-0.5 * t))


def test_velocity_acf_theory_at_zero():
    t = np.array([0.0])
    result = velocity_acf_theory(t, v=2.0, D_R=0.5)
    assert abs(result[0] - 4.0) < 1e-10   # v² * exp(0) = 4


def test_velocity_acf_theory_decays():
    t = np.linspace(0, 10, 100)
    result = velocity_acf_theory(t, v=2.0, D_R=0.5)
    np.testing.assert_allclose(result, 4.0 * np.exp(-0.5 * t))
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_theory.py -k "acf_theory" -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add functions to simulation/theory.py**

Append to the bottom of `simulation/theory.py`:
```python

def orientation_acf_theory(t: np.ndarray, D_R: float) -> np.ndarray:
    """Theoretical orientation ACF for ABP: exp(-D_R * t)."""
    return np.exp(-D_R * t)


def velocity_acf_theory(t: np.ndarray, v: float, D_R: float) -> np.ndarray:
    """Theoretical velocity ACF for ABP (lag > 0): v² * exp(-D_R * t)."""
    return v**2 * np.exp(-D_R * t)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_theory.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add simulation/theory.py tests/test_theory.py
git commit -m "feat: add orientation_acf_theory and velocity_acf_theory to theory.py"
```

---

## Task 5: Plotter — plot_correlations

**Files:**
- Modify: `visualization/plotter.py`

No new test file for the plotter (matplotlib output is visual; correctness verified by running main.py in Task 6).

- [ ] **Step 1: Add plot_correlations to visualization/plotter.py**

Append to the bottom of `visualization/plotter.py`:
```python

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
```

- [ ] **Step 2: Commit**

```
git add visualization/plotter.py
git commit -m "feat: add plot_correlations to plotter"
```

---

## Task 6: compare_boundary_conditions + main.py demo

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add compare_boundary_conditions and demo calls to main.py**

Replace the entire contents of `main.py`:
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf, velocity_acf, position_acf
from simulation.theory import (
    passive_msd, active_msd, effective_diffusion, rotational_relaxation_time,
    orientation_acf_theory, velocity_acf_theory,
)
from visualization.plotter import plot_trajectory, plot_msd, plot_correlations

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

    fig2, ax = plt.subplots(figsize=(7, 5))
    plot_msd(passive_trajs, DT_SIM, label="Passive (Eq. 3)",             ax=ax)
    plot_msd(active_trajs,  DT_SIM, label=f"Active v={V} µm/s (Eq. 4)", ax=ax)
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

    # ── 4. Correlation functions ───────────────────────────────────────────────
    max_lag = 300
    dt_arr  = np.arange(max_lag) * DT_SIM

    lag_steps_o, acf_o = orientation_acf(active_trajs, max_lag=max_lag)
    lag_steps_v, acf_v = velocity_acf(active_trajs, dt=DT_SIM, max_lag=max_lag)
    lag_steps_r, acf_r = position_acf(active_trajs, max_lag=max_lag)

    acfs = {
        "orientation": (dt_arr, acf_o),
        "velocity":    (dt_arr, acf_v),
        "position":    (dt_arr, acf_r),
    }
    theory_curves = {
        "orientation": orientation_acf_theory(dt_arr, D_R),
        "velocity":    velocity_acf_theory(dt_arr, V, D_R),
    }

    fig4 = plot_correlations(acfs, theory=theory_curves)
    fig4.savefig("correlations.png", dpi=150)
    print("Saved correlations.png")

    plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run main.py and verify output**

```
python main.py
```
Expected console output:
```
Saved trajectories.png
tau_R = 6.25 s    D_eff = 12.72 µm²/s
Saved msd_comparison.png
Saved boundary_comparison.png
Saved correlations.png
```
Four figure windows open. Check:
- `boundary_comparison.png`: top row shows 4 trajectory plots with dashed box boundary visible for reflect/stop/slip; bottom row shows MSD log-log with `"none"` growing indefinitely and bounded modes plateauing.
- `correlations.png`: orientation ACF decays as `exp(−D_R t)` matching the dashed theory line; velocity ACF similar shape scaled by `v²`; position ACF grows monotonically.

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```
git add main.py
git commit -m "feat: add compare_boundary_conditions and correlation function demo to main.py"
```

---

## Task 7: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md**

Apply the following changes to `README.md`:

**1. Update "Current phase" line** (line 5):
```
**Current phase:** Phase 1 — single-particle passive and active Brownian motion with boundary conditions and correlation analysis.
```

**2. After the "Typical physical parameters" block, add a new section:**
```markdown
### Physical D_T / D_R (Stokes-Einstein-Debye)

Instead of setting `D_T` and `D_R` manually, derive them from the particle radius using the Stokes-Einstein-Debye relations:

```python
params = SimParams.from_physical(radius_um=1.0, T_K=300.0, eta_Pa_s=1e-3, v=2.0)
# → D_T ≈ 0.214 µm²/s,  D_R ≈ 0.161 rad²/s
```

### Boundary Conditions

Set `boundary` and `box_size` in `SimParams` to confine the particle to a square box `[−L, L] × [−L, L]`:

| `boundary` | Behaviour |
|---|---|
| `"none"` (default) | Unbounded — particle roams freely |
| `"reflect"` | Elastic reflection; orientation φ also mirrors at the wall |
| `"stop"` | Particle clamps to the wall; resumes on next valid step |
| `"slip"` | Motion component parallel to wall is preserved; perpendicular component is zeroed |

```python
params = SimParams(boundary="reflect", box_size=5.0, v=2.0, ...)
fig = compare_boundary_conditions(ActiveBrownianParticle, params, box_size=5.0)
```

### Correlation Functions

Three autocorrelation functions available in `simulation/analysis.py`:

| Function | Formula | Theory (ABP) |
|---|---|---|
| `orientation_acf` | `⟨cos(φ(t+τ) − φ(t))⟩` | `exp(−D_R τ)` |
| `velocity_acf` | `⟨v(t+τ)·v(t)⟩` from position differences | `v²·exp(−D_R τ)` at τ>0 |
| `position_acf` | `⟨r(t)·r(t+τ)⟩` | no closed form |
```

**3. Update the "How to Run" outputs block** to mention the two new output files:
```
Produces:
- `trajectories.png` — passive (left) and active (right) trajectories, colored blue→red over time
- `msd_comparison.png` — log-log MSD showing active particle's ballistic-to-diffusive crossover
- `boundary_comparison.png` — trajectory + MSD comparison across all four boundary modes
- `correlations.png` — orientation, velocity, and position ACFs with theory overlays
```

**4. Update the File Structure section** to add new files:
```
├── simulation/
│   ├── params.py         # SimParams dataclass — all tunable parameters + from_physical classmethod
│   ├── particle.py       # Particle ABC with boundary logic; PassiveBrownianParticle; ActiveBrownianParticle
│   ├── simulator.py      # Simulator — time-steps a particle, returns trajectory array
│   ├── analysis.py       # orientation_acf, velocity_acf, position_acf
│   └── theory.py         # Analytical MSD + ACF formulas for validation
├── visualization/
│   └── plotter.py        # plot_trajectory(), plot_msd(), plot_correlations()
├── tests/
│   ├── test_params.py    # Tests for SimParams boundary validation and from_physical
│   ├── test_particle.py  # Unit tests for all particle types and boundary modes
│   ├── test_simulator.py # Integration tests for Simulator (shape, determinism, MSD scaling)
│   ├── test_theory.py    # Unit tests for all analytical formulas in theory.py
│   └── test_analysis.py  # Tests for correlation function computation
```

- [ ] **Step 2: Commit**

```
git add README.md
git commit -m "docs: update README with boundary conditions, from_physical, and correlation functions"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| boundary="reflect" — mirror position + reflect φ | Task 2 |
| boundary="stop" — clamp to wall | Task 2 |
| boundary="slip" — project to parallel component | Task 2 |
| `box_size=None` + non-"none" boundary raises ValueError | Task 1 |
| `from_physical` classmethod with Eq. 1 & 2 | Task 1 |
| orientation_acf | Task 3 |
| velocity_acf (finite-difference) | Task 3 |
| position_acf | Task 3 |
| orientation_acf_theory, velocity_acf_theory | Task 4 |
| plot_correlations with theory overlay | Task 5 |
| compare_boundary_conditions (2×4 figure) | Task 6 |
| README updated | Task 7 |

### Placeholder scan
No TBDs, no "add error handling later", no "similar to Task N". All code blocks are complete.

### Type consistency
- `_apply_boundary(x_new, y_new, phi) -> tuple[float, float, float]` — called identically in both `step()` implementations ✓
- `orientation_acf / velocity_acf / position_acf` all return `(np.ndarray, np.ndarray)` — consumed identically in `plot_correlations` via `acfs[key]` unpacking ✓
- `plot_correlations(acfs, theory, ax_array)` — `acfs` dict keys `"orientation"`, `"velocity"`, `"position"` match the `configs` list in the function body ✓
- `from_physical` returns `cls(D_T=D_T, D_R=D_R, **kwargs)` — `kwargs` passes through to `SimParams.__init__`, which triggers `__post_init__` validation ✓
- `compare_boundary_conditions` uses `SimParams(boundary=mode, box_size=box_size if mode != "none" else None)` — consistent with the `__post_init__` guard ✓
