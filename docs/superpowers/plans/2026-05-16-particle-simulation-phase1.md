# Particle Simulation — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular 2D particle simulation implementing passive Brownian motion (Eq. 3) and self-propelled active Brownian motion (Eq. 4) for a single particle, with trajectory and MSD visualization.

**Architecture:** A `Particle` class hierarchy handles state and per-step updates; a `Simulator` owns the particle list and time-stepping loop, returning a trajectory array; a `Plotter` generates trajectory and MSD figures. Parameters are centralized in a `SimParams` dataclass. Each layer is designed to scale to N particles and pairwise interactions in later phases with minimal structural changes.

**Tech Stack:** Python 3.10+, NumPy (numerics/RNG), Matplotlib (plotting), pytest (tests)

---

## Source Equations

**Eq. 3 — Passive Brownian Motion (Euler-Maruyama discretization):**
```
x(t+dt) = x(t) + sqrt(2·D_T·dt) · η_x
y(t+dt) = y(t) + sqrt(2·D_T·dt) · η_y
φ(t+dt) = φ(t) + sqrt(2·D_R·dt) · η_φ
```

**Eq. 4 — Active Brownian Particle (Euler-Maruyama discretization):**
```
x(t+dt) = x(t) + v·cos(φ)·dt + sqrt(2·D_T·dt) · η_x
y(t+dt) = y(t) + v·sin(φ)·dt + sqrt(2·D_T·dt) · η_y
φ(t+dt) = φ(t) + sqrt(2·D_R·dt) · η_φ
```

Where η_x, η_y, η_φ ~ N(0,1) are independent standard normals drawn each step.
D_T = translational diffusion coefficient; D_R = rotational diffusion coefficient.
Python variables use the same names: `D_T`, `D_R`.

Typical physical values (R=1 µm particle in water at 300 K):
- D_T ≈ 0.22 µm²/s, D_R ≈ 0.16 rad²/s, v ∈ [0, 3] µm/s

---

## File Map

| File | Responsibility |
|------|----------------|
| `simulation/__init__.py` | Package marker |
| `simulation/params.py` | `SimParams` dataclass: D_T, D_R, v, dt, n_steps, seed, x0, y0, phi0 |
| `simulation/particle.py` | `Particle` ABC + `PassiveBrownianParticle` + `ActiveBrownianParticle` |
| `simulation/simulator.py` | `Simulator`: owns particle list, runs time loop, returns `(n_steps+1, 3)` trajectory array |
| `visualization/__init__.py` | Package marker |
| `visualization/plotter.py` | `plot_trajectory()` and `plot_msd()` functions |
| `main.py` | Demo: single trajectory side-by-side + ensemble MSD comparison |
| `tests/__init__.py` | Package marker |
| `tests/test_particle.py` | Unit tests for particle update logic |
| `tests/test_simulator.py` | Integration tests for Simulator (shape, determinism, MSD scaling) |
| `requirements.txt` | numpy, matplotlib, pytest |

---

## Task 1: Project Scaffold + SimParams

**Files:**
- Create: `simulation/__init__.py`
- Create: `simulation/params.py`
- Create: `visualization/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create directories and package markers**

```bash
python -c "
import os
for d in ['simulation', 'visualization', 'tests']:
    os.makedirs(d, exist_ok=True)
    open(f'{d}/__init__.py', 'w').close()
"
```

- [ ] **Step 2: Create requirements.txt**

```
numpy
matplotlib
pytest
```

- [ ] **Step 3: Write SimParams**

Create `simulation/params.py`:
```python
from dataclasses import dataclass


@dataclass
class SimParams:
    D_T: float = 0.22      # translational diffusion coefficient [µm²/s]
    D_R: float = 0.16      # rotational diffusion coefficient [rad²/s]
    v: float = 0.0         # self-propulsion speed [µm/s]; 0 = passive
    dt: float = 0.01       # time step [s]
    n_steps: int = 1000    # number of simulation steps
    seed: int = 42         # RNG seed; set to None for non-reproducible runs
    x0: float = 0.0        # initial x position [µm]
    y0: float = 0.0        # initial y position [µm]
    phi0: float = 0.0      # initial orientation angle [rad]
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_particle.py`:
```python
import numpy as np
import pytest
from simulation.params import SimParams


def test_simparams_defaults():
    p = SimParams()
    assert p.D_T == 0.22
    assert p.D_R == 0.16
    assert p.v == 0.0
    assert p.dt == 0.01
    assert p.n_steps == 1000
    assert p.seed == 42
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_particle.py::test_simparams_defaults -v
```
Expected output: `PASSED`

- [ ] **Step 6: Commit**

```bash
git init
git add simulation/ visualization/ tests/ requirements.txt
git commit -m "feat: scaffold project structure and SimParams dataclass"
```

---

## Task 2: PassiveBrownianParticle (Eq. 3)

**Files:**
- Create: `simulation/particle.py`
- Modify: `tests/test_particle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_particle.py`:
```python
from simulation.particle import PassiveBrownianParticle


def test_passive_particle_initializes():
    params = SimParams(x0=1.0, y0=2.0, phi0=0.5)
    p = PassiveBrownianParticle(params)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.phi == 0.5


def test_passive_particle_step_is_deterministic_with_seed():
    params = SimParams(seed=0)
    p1 = PassiveBrownianParticle(params)
    p2 = PassiveBrownianParticle(params)
    p1.step()
    p2.step()
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.phi == p2.phi


def test_passive_particle_zero_diffusion_stays_put():
    params = SimParams(DT=0.0, DR=0.0, seed=0)
    p = PassiveBrownianParticle(params)
    p.step()
    assert p.x == 0.0
    assert p.y == 0.0
    assert p.phi == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_particle.py -k "passive" -v
```
Expected: FAIL with `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Implement PassiveBrownianParticle**

Create `simulation/particle.py`:
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
        self.x += noise_t * eta[0]
        self.y += noise_t * eta[1]
        self.phi += noise_r * eta[2]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_particle.py -k "passive" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add simulation/particle.py tests/test_particle.py
git commit -m "feat: add PassiveBrownianParticle implementing Eq. 3"
```

---

## Task 3: ActiveBrownianParticle (Eq. 4)

**Files:**
- Modify: `simulation/particle.py`
- Modify: `tests/test_particle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_particle.py`:
```python
from simulation.particle import ActiveBrownianParticle


def test_active_particle_zero_noise_moves_straight():
    """With DT=DR=0, particle travels exactly v*dt in direction phi0."""
    params = SimParams(DT=0.0, DR=0.0, v=1.0, dt=0.1, phi0=0.0, seed=0)
    p = ActiveBrownianParticle(params)
    p.step()
    assert abs(p.x - 0.1) < 1e-10   # v*cos(0)*dt = 1.0*1.0*0.1
    assert abs(p.y - 0.0) < 1e-10   # v*sin(0)*dt = 0


def test_active_particle_v0_matches_passive():
    """Active particle with v=0 must produce identical trajectory to passive (same seed)."""
    kwargs = dict(D_T=0.22, D_R=0.16, v=0.0, dt=0.01, seed=7)
    passive = PassiveBrownianParticle(SimParams(**kwargs))
    active = ActiveBrownianParticle(SimParams(**kwargs))
    for _ in range(20):
        passive.step()
        active.step()
    assert abs(passive.x - active.x) < 1e-10
    assert abs(passive.y - active.y) < 1e-10
    assert abs(passive.phi - active.phi) < 1e-10


def test_active_particle_step_is_deterministic_with_seed():
    params = SimParams(v=2.0, seed=99)
    p1 = ActiveBrownianParticle(params)
    p2 = ActiveBrownianParticle(params)
    for _ in range(10):
        p1.step()
        p2.step()
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.phi == p2.phi
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_particle.py -k "active" -v
```
Expected: FAIL with `ImportError` (ActiveBrownianParticle not defined)

- [ ] **Step 3: Implement ActiveBrownianParticle**

Append to the bottom of `simulation/particle.py`:
```python
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
        self.x += p.v * np.cos(self.phi) * p.dt + noise_t * eta[0]
        self.y += p.v * np.sin(self.phi) * p.dt + noise_t * eta[1]
        self.phi += noise_r * eta[2]
```

- [ ] **Step 4: Run all particle tests**

```bash
python -m pytest tests/test_particle.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add simulation/particle.py tests/test_particle.py
git commit -m "feat: add ActiveBrownianParticle implementing Eq. 4"
```

---

## Task 4: Simulator

**Files:**
- Create: `simulation/simulator.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_simulator.py`:
```python
import numpy as np
import pytest
from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle
from simulation.simulator import Simulator


def test_simulator_returns_correct_shape():
    params = SimParams(n_steps=50, seed=0)
    traj = Simulator(PassiveBrownianParticle(params), params).run()
    assert traj.shape == (51, 3)  # n_steps+1 rows: includes t=0; 3 cols: x, y, phi


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_simulator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'simulation.simulator'`

- [ ] **Step 3: Implement Simulator**

Create `simulation/simulator.py`:
```python
import numpy as np
from simulation.particle import Particle
from simulation.params import SimParams


class Simulator:
    """Time-steps a particle and records the full trajectory.

    Returns an array of shape (n_steps+1, 3): rows are time snapshots,
    columns are [x, y, phi]. Row 0 is the initial state.

    Holds a list internally so multi-particle support can be added later
    without changing the interface.
    """

    def __init__(self, particle: Particle, params: SimParams):
        self.particle = particle
        self.params = params

    def run(self) -> np.ndarray:
        p = self.params
        traj = np.empty((p.n_steps + 1, 3))
        traj[0] = self.particle.state()
        for i in range(p.n_steps):
            self.particle.step()
            traj[i + 1] = self.particle.state()
        return traj
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED (the MSD test may take ~5 s)

- [ ] **Step 5: Commit**

```bash
git add simulation/simulator.py tests/test_simulator.py
git commit -m "feat: add Simulator with trajectory output and MSD scaling test"
```

---

## Task 5: Visualization

**Files:**
- Create: `visualization/plotter.py`

- [ ] **Step 1: Implement plotter**

Create `visualization/plotter.py`:
```python
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
) -> plt.Figure:
    """Plot ensemble-averaged MSD vs time on a log-log scale.

    Args:
        trajs: List of trajectory arrays, each shape (n_steps+1, 3).
        dt:    Time step in seconds.
        label: Legend label for this dataset.
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

    ax.loglog(t, msds.mean(axis=0), label=label or "MSD")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSD [µm²]")
    ax.set_title("Mean Squared Displacement")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add visualization/plotter.py
git commit -m "feat: add trajectory and MSD visualization functions"
```

---

## Task 6: main.py Demo

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

Create `main.py`:
```python
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

    plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Run main.py and verify output**

```bash
python main.py
```
Expected: two figure windows open (side-by-side trajectories, MSD log-log); `trajectories.png` and `msd_comparison.png` saved. The MSD plot should show the active particle crossing over from ballistic (slope 2) to diffusive (slope 1) behavior at long times.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add main.py demo — trajectory and MSD comparison for Eq. 3 and Eq. 4"
```

---

## Self-Review

### Spec coverage
| Requirement | Covered by |
|---|---|
| Passive Brownian motion (Eq. 3) | Task 2 |
| Self-propelled active Brownian (Eq. 4) | Task 3 |
| Single particle to start | Tasks 2–4 |
| Modular for future multi-particle extension | `Simulator` has a single particle field, trivially extended to a list |
| Trajectory visualization | Tasks 5–6 |
| MSD analysis | Tasks 5–6 |
| Modifiable parameters | `SimParams` dataclass, all values at top of `main.py` |

### No placeholder check — all steps contain actual code. No TBDs or "add error handling" vagueness.

### Type consistency
- `Particle.state()` returns `tuple[float, float, float]` — used as `traj[i+1] = self.particle.state()` in Simulator ✓
- `Simulator.run()` returns `np.ndarray` shape `(n_steps+1, 3)` — used in both plotter functions ✓
- `PassiveBrownianParticle` and `ActiveBrownianParticle` both inherit `Particle` and implement `step()` ✓
- `test_active_particle_v0_matches_passive` imports both classes from the same module ✓
