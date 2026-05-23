# Design: Boundary Conditions, Physical Diffusion Mode, and Correlation Functions

**Date:** 2026-05-23
**Status:** Approved

---

## Overview

Three additions to the particle simulation:
1. Three boundary condition modes (reflection, stop, slip)
2. A physical mode that derives D_T and D_R from the Stokes-Einstein-Debye relations
3. Three correlation functions (orientation, velocity, position) with theory overlays

---

## 1. Boundary Conditions

### Parameters

`SimParams` gains two new fields:

```python
boundary: str = "none"       # one of: "none", "reflect", "stop", "slip"
box_size: float | None = None  # half-width L in µm; box = [−L, L] × [−L, L]
```

`boundary="none"` and `box_size=None` reproduce existing behavior with zero changes to
any downstream code.

### Logic

A `_apply_boundary(x_new, y_new, phi) -> (x, y, phi)` method is added to the `Particle`
base class. Every `step()` implementation calls it before committing the new position.
The method dispatches on `self.params.boundary`.

**Reflect**

Mirror the position back across the wall. Applied axis-by-axis so corner hits are handled:

```
if x_new > +L:  x_new = 2L − x_new;  phi_x_reflected = True
if x_new < −L:  x_new = −2L − x_new; phi_x_reflected = True
if y_new > +L:  y_new = 2L − y_new;  phi_y_reflected = True
if y_new < −L:  y_new = −2L − y_new; phi_y_reflected = True
```

Orientation reflection (applies to all particle types for physical correctness):
- x-wall hit: `φ → π − φ`
- y-wall hit: `φ → −φ`
- corner hit: `φ → π + φ`

**Stop (clamp)**

```
x_new = clip(x_new, −L, L)
y_new = clip(y_new, −L, L)
φ unchanged
```

The particle parks at the wall. Future steps draw fresh noise and may move inward
naturally. No special resampling loop; statistics emerge from repeated clamping.

**Slip (projection)**

Zero out only the component(s) that cross the boundary; keep the other:

```
dx = x_new − x_old
dy = y_new − y_old
if x_new exits boundary: dx = 0
if y_new exits boundary: dy = 0
x_new = x_old + dx
y_new = y_old + dy
φ unchanged
```

If both components exit, the particle stays put for that step.

### Invariants

- `boundary="none"` is the default; all existing tests remain valid without modification.
- `box_size=None` with any non-"none" boundary raises `ValueError` at construction time.
- The initial position `(x0, y0)` is not validated against `box_size`; caller's responsibility.

---

## 2. Physical D_T / D_R Mode

### Equations (Stokes-Einstein-Debye, sphere of radius R)

```
D_T = k_B T / (6π η R)    [µm²/s]   — Eq. 1
D_R = k_B T / (8π η R³)   [rad²/s]  — Eq. 2
```

k_B = 1.380649 × 10⁻²³ J/K. R given in µm, converted to m internally.

### Interface

A classmethod on `SimParams`:

```python
@classmethod
def from_physical(
    cls,
    radius_um: float,
    T_K: float = 300.0,
    eta_Pa_s: float = 1e-3,
    **kwargs,          # all other SimParams fields (v, dt, n_steps, seed, ...)
) -> "SimParams":
```

Returns a normal `SimParams` with `D_T` and `D_R` set to the computed values. No boolean
flag; no new code path in `Particle` or `Simulator`. Callers can inspect the returned
object's `D_T` and `D_R` for verification.

**Reference values** (R=1 µm, water at 300 K, η=1×10⁻³ Pa·s):
- D_T ≈ 0.214 µm²/s
- D_R ≈ 0.161 rad²/s

These are consistent with the empirical defaults already in `SimParams`.

---

## 3. Correlation Functions

### New file: `simulation/analysis.py`

Three functions, all with the same signature pattern:

```python
def orientation_acf(trajs, max_lag=None) -> tuple[np.ndarray, np.ndarray]
def velocity_acf(trajs, dt, max_lag=None) -> tuple[np.ndarray, np.ndarray]
def position_acf(trajs, max_lag=None) -> tuple[np.ndarray, np.ndarray]
```

- `trajs`: list of trajectory arrays, each shape `(N+1, 3)` with columns `[x, y, phi]`
- `max_lag`: number of lag steps to compute (default: `N//2`)
- Returns `(lag_steps, mean_acf)` — ensemble average over all trajectories, and for each
  trajectory a time-average over all valid starting points

**Orientation ACF**

```
C_φ(τ) = ⟨cos(φ(t+τ) − φ(t))⟩_{t, ensemble}
```

Analytical prediction (ABP): `exp(−D_R τ)`

**Velocity ACF**

Computed from finite-difference velocities: `v(t) = (r(t+dt) − r(t)) / dt`

```
C_v(τ) = ⟨v(t+τ) · v(t)⟩_{t, ensemble}
```

For ABP analytically: `v²·exp(−D_R τ)` at lags τ > 0; at τ=0 it includes the noise
variance `2D_T/dt`, making the zero-lag spike much larger than subsequent values.

**Position ACF**

```
C_r(τ) = ⟨r(t) · r(t+τ)⟩_{t, ensemble}
```

No clean analytical form for free diffusion (grows with t). Useful for visualizing
positional memory and comparing passive vs. active regimes qualitatively.

### Theory overlays: `simulation/theory.py`

Two new functions:
- `orientation_acf_theory(t, D_R)` → `exp(−D_R t)`
- `velocity_acf_theory(t, v, D_R)` → `v² · exp(−D_R t)` (valid for τ > 0)

### Plotting: `visualization/plotter.py`

New function:

```python
def plot_correlations(
    acfs: dict[str, tuple[np.ndarray, np.ndarray]],  # name → (lag_times, values)
    theory: dict[str, np.ndarray] | None = None,
    ax_array: np.ndarray | None = None,
) -> plt.Figure
```

Produces a 1×3 subplot figure (one panel per correlation function). Theory curves
overlaid as dashed lines where provided. x-axis in seconds.

---

## File Changes Summary

| File | Change |
|---|---|
| `simulation/params.py` | Add `boundary`, `box_size` fields; add `from_physical` classmethod; add `__post_init__` validation |
| `simulation/particle.py` | Add `_apply_boundary` to `Particle` base; call it in both `step()` implementations |
| `simulation/analysis.py` | New file: `orientation_acf`, `velocity_acf`, `position_acf` |
| `simulation/theory.py` | Add `orientation_acf_theory`, `velocity_acf_theory` |
| `visualization/plotter.py` | Add `plot_correlations` |
| `tests/test_particle.py` | Tests for all three boundary modes (reflect, stop, slip) |
| `tests/test_analysis.py` | Tests for correlation functions (shape, symmetry, passive limit) |
| `tests/test_params.py` | Tests for `from_physical` (unit check, reference values) |
| `main.py` | Demo: boundary condition examples + correlation function plots |
| `README.md` | Update with new features |

---

## Constraints and Non-Goals

- Box is always square and centered at the origin. Rectangular or offset boxes are not in scope.
- No multi-particle interactions; boundary logic is per-particle.
- No 3D extension in this phase.
- Correlation functions are computed post-hoc from saved trajectories; no online accumulation.
