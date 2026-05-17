# Cell Migration Particle Simulation

A modular 2D particle simulation built incrementally across weekly assignments. Each phase adds new physics on top of a clean, testable foundation.

**Current phase:** Phase 1 — single-particle passive and active Brownian motion.

---

## What This Simulates

Two physics models from *Active Brownian Particles* (Romanczuk et al.):

### Eq. 3 — Passive Brownian Motion
A particle undergoing pure thermal diffusion. Each step:
```
x(t+dt) = x(t) + sqrt(2·D_T·dt) · η_x
y(t+dt) = y(t) + sqrt(2·D_T·dt) · η_y
φ(t+dt) = φ(t) + sqrt(2·D_R·dt) · η_φ
```
`η_x, η_y, η_φ ~ N(0,1)` independent noise each step.

### Eq. 4 — Active Brownian Particle (Self-Propelled)
Same as Eq. 3 but the particle also propels itself in direction `φ` at speed `v`:
```
x(t+dt) = x(t) + v·cos(φ)·dt + sqrt(2·D_T·dt) · η_x
y(t+dt) = y(t) + v·sin(φ)·dt + sqrt(2·D_T·dt) · η_y
φ(t+dt) = φ(t) + sqrt(2·D_R·dt) · η_φ
```
At short times: ballistic (straight line). At long times (t >> 1/D_R): diffusive again.

**Typical physical parameters** (1 µm particle in water at 300 K):
- `D_T` ≈ 0.22 µm²/s — translational diffusion
- `D_R` ≈ 0.16 rad²/s — rotational diffusion
- `v` ∈ [0, 3] µm/s — self-propulsion speed

---

## Theoretical Predictions (simulation/theory.py)

The MSD has exact analytical solutions used to validate the simulation:

| Quantity | Formula |
|----------|---------|
| Passive MSD | `4·D_T·t` (slope 1 always) |
| Active MSD (exact) | `4·D_T·t + (2v²/D_R)·[t − (1−e^{−D_R t})/D_R]` |
| Short-time asymptote (t ≪ τ_R) | `4·D_T·t + v²·t²` — ballistic, **slope 2** |
| Long-time asymptote (t ≫ τ_R) | `4·D_eff·t` — diffusive, **slope 1** |
| Crossover time | `τ_R = 1/D_R` |
| Enhanced diffusivity | `D_eff = D_T + v²/(2·D_R)` |

With default params: `τ_R = 6.25 s`, `D_eff = 12.72 µm²/s`.

The MSD plot (`msd_comparison.png`) overlays all theoretical curves on the simulation data, with a vertical dashed line at `τ_R`.

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Save trajectory + MSD plots to PNG files
```bash
python main.py
```
Produces:
- `trajectories.png` — passive (left) and active (right) trajectories, colored blue→red over time
- `msd_comparison.png` — log-log MSD showing active particle's ballistic-to-diffusive crossover

### Live animation
```bash
python animate.py
```
Opens a side-by-side window: passive particle (blue) on left, active particle (red + orientation arrow) on right. Plays back in real time.

### Run tests
```bash
python -m pytest tests/ -v
```

---

## File Structure

```
code/
├── simulation/
│   ├── params.py         # SimParams dataclass — all tunable parameters in one place
│   ├── particle.py       # Particle ABC, PassiveBrownianParticle, ActiveBrownianParticle
│   ├── simulator.py      # Simulator — time-steps a particle, returns trajectory array
│   └── theory.py         # Analytical MSD formulas for validation
├── visualization/
│   └── plotter.py        # plot_trajectory(), plot_msd() — static figure functions
├── tests/
│   ├── test_particle.py  # Unit tests for PassiveBrownianParticle and ActiveBrownianParticle
│   ├── test_simulator.py # Integration tests for Simulator (shape, determinism, MSD scaling)
│   └── test_theory.py    # Unit tests for all analytical formulas in theory.py
├── main.py               # Entry point — saves trajectories.png and msd_comparison.png
├── animate.py            # Live animation entry point
└── requirements.txt      # numpy, matplotlib, pytest
```

### File responsibilities in detail

| File | What it does |
|------|-------------|
| `simulation/params.py` | Single source of truth for all physics parameters. Change `D_T`, `D_R`, `v`, `dt`, `n_steps`, `seed`, `x0`, `y0`, `phi0` here. |
| `simulation/particle.py` | `Particle` is an abstract base class. Each subclass implements `step()` for one physics model. Adding a new model = add a new subclass. |
| `simulation/simulator.py` | `Simulator(particle, params).run()` returns a `(n_steps+1, 3)` NumPy array where columns are `[x, y, phi]`. Row 0 is the initial state. **Note:** `run()` mutates the particle in place — create a fresh particle for each run. |
| `simulation/theory.py` | Pure-function analytical predictions: `passive_msd`, `active_msd`, `active_msd_short_time`, `active_msd_long_time`, `effective_diffusion`, `rotational_relaxation_time`. No simulation state. |
| `visualization/plotter.py` | `plot_trajectory(traj, title, ax)` draws a time-colored path. `plot_msd(trajs, dt, label, ax, theory_curves)` draws ensemble-averaged MSD on a log-log scale with optional theoretical overlays. |
| `main.py` | Demo script. Top of file has all constants (`D_T`, `D_R`, `V`, etc.) — edit there to experiment. Prints `tau_R` and `D_eff` on startup. |
| `animate.py` | Same constants at the top. `SKIP` controls playback speed (higher = faster). `TRAIL` controls how many past positions are shown. |

---

## Key Design Decisions

- **Per-particle RNG:** Each `Particle` instance owns its own `np.random.default_rng(seed)`. This makes runs reproducible and ensemble averaging correct (each particle is independent).
- **`SimParams` as the single knob:** To change physics, edit one dataclass instance. Nothing is hardcoded inside `Particle` or `Simulator`.
- **Trajectory as NumPy array:** `(n_steps+1, 3)` with columns `[x, y, phi]`. Designed to extend to `(N_particles, n_steps+1, 3)` in the multi-particle phase.
- **`Particle` ABC:** Adding a new model (e.g. run-and-tumble, chiral swimmer) = write a new subclass and implement `step()`. Nothing else changes.

---

## Tests (18 total)

| Test | What it verifies |
|------|-----------------|
| `test_simparams_defaults` | All 9 SimParams fields have correct defaults |
| `test_passive_particle_initializes` | x, y, phi set from params |
| `test_passive_particle_step_is_deterministic_with_seed` | Same seed → same trajectory |
| `test_passive_particle_zero_diffusion_stays_put` | D_T=D_R=0 → particle doesn't move |
| `test_active_particle_zero_noise_moves_straight` | D_T=D_R=0, v=1, phi=0 → x advances by v·dt exactly |
| `test_active_particle_v0_matches_passive` | Active with v=0 equals passive bit-for-bit |
| `test_active_particle_step_is_deterministic_with_seed` | Same seed → same trajectory |
| `test_simulator_returns_correct_shape` | Output shape is (n_steps+1, 3) |
| `test_simulator_first_row_is_initial_state` | Row 0 matches x0, y0, phi0 |
| `test_simulator_is_deterministic` | 3 independent runs with same seed produce identical arrays |
| `test_passive_msd_scales_linearly_with_time` | Ensemble MSD ≈ 4·D_T·t within 20% (200 realizations) |
| `test_passive_msd_linear` | passive_msd = 4·D_T·t exactly |
| `test_active_msd_reduces_to_passive_when_v0` | active_msd with v=0 equals passive_msd |
| `test_active_msd_short_time_ballistic` | active_msd ≈ 4·D_T·t + v²t² for t ≪ τ_R |
| `test_active_msd_long_time_slope` | d(MSD)/dt → 4·D_eff for t ≫ τ_R |
| `test_active_msd_short_time_matches_asymptote` | active_msd_short_time = 4·D_T·t + v²t² exactly |
| `test_effective_diffusion` | D_eff = D_T + v²/(2·D_R) |
| `test_rotational_relaxation_time` | τ_R = 1/D_R |

---

## Future Phases

| Phase | What will be added |
|-------|-------------------|
| Phase 2 | N particles in one simulation; multi-particle `Simulator` |
| Phase 3 | Particle-particle interactions (e.g. steric repulsion, alignment) |
| Phase 4+ | TBD based on weekly assignments |

---

## References

- Romanczuk et al., *Active Brownian Particles*, EPJ Special Topics (2012) — equations 3 & 4, page 7
- Sed Boker Jan 2026 lecture notes — overall modeling context
