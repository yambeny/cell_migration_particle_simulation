# Cell Migration Particle Simulation

A modular 2D particle simulation built incrementally across weekly assignments. Each phase adds new physics on top of a clean, testable foundation.

**Current phase:** Phase 1 — single-particle passive and active Brownian motion, boundary conditions, and correlation analysis.

**Exploratory:** `main.py` is now calibrated to hPSC parameters (see `docs/hPSC_parameters.md`), and a set of standalone scripts explore extensions of the ABP model toward reproducing measured endoderm migration statistics (see `docs/endoderm_migration_models.md`).

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

### Physical D_T / D_R (Stokes-Einstein-Debye)

Instead of setting `D_T` and `D_R` manually, derive them from the particle radius using the Stokes-Einstein-Debye relations (Eq. 1 & 2):

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
| `"slip"` | Motion component parallel to wall is preserved; perpendicular is zeroed |

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
| Local MSD exponent | `α(t) = d(log MSD)/d(log t)` — `active_msd_exponent()`; α→1 as t→0 and t→∞, peaks near `τ_R` |

With default (hPSC) params: `τ_R ≈ 1613 s`, `D_eff ≈ D_T` (Pe < 1, so the active MSD only shows a soft α peak slightly above 1).

The MSD plot (`msd_comparison.png`) is now a two-panel figure: ensemble MSD (log-log) on top, and the local exponent `α(t)` for passive vs. active theory on the bottom, with the global power-law fit and the peak α marked.

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
- `msd_comparison.png` — two-panel: ensemble MSD (log-log, sim + exact theory + long-time asymptote) and local exponent α(t) (passive vs. active theory, with peak α and global power-law fit)
- `boundary_comparison_passive.png` — trajectory + MSD for all four boundary modes (passive particle)
- `boundary_comparison_active.png` — same for active particle; MSD shows free-space theory + reflect saturation (2L²/3)
- `correlations.png` — orientation ACF with theory (input D_R) and nonlinear fit (recovered D_R) overlays, plus τ_R/τ_c/persistence-length annotations
- `stop_vs_slip_demo.png` — x-position density histograms at extreme parameters to visualise stop vs slip difference
- `msd_comparison_theory.png` — pure-theory comparison of MSD and α(t) for Stokes-Einstein vs. biological (Sde Boker) D_R/v

### Live animation
```bash
python animate.py
```
Opens a side-by-side window: passive particle (blue) on left, active particle (red + orientation arrow) on right. Plays back in real time.

### Exploratory: Endoderm Migration Models
```bash
python abp_cil.py            # ABP + steric repulsion + contact-inhibition-of-locomotion
python active_fbm.py          # Active particle with fractional Gaussian noise (translation), MSD ~ t^2H
python active_fbm_angular.py  # ...with correlated (fGn) angular noise to shape the DACF
python hetero_abp.py          # Heterogeneous ABP: log-normal distribution of τ_R
python hetero_abp_pareto.py   # Heterogeneous ABP: Pareto-distributed τ_R → exact power-law DACF
```
Each script saves its plot(s) to `other images/`. These explore why the standard ABP
(exponential DACF, returns to MSD slope α=1 at long times) cannot reproduce the
measured endoderm migration statistics (DACF ~ τ⁻⁰·⁸³, MSD ~ t^1.4, no crossover —
see `docs/endoderm_migration_models.md`), and what model changes are needed to get there.
`docs/pdf_pages/` contains the source lecture-slide images (Sde Boker, Jan 2026) these
numbers are taken from.

### Run tests
```bash
python -m pytest tests/ -v
```

---

## File Structure

```
code/
├── simulation/
│   ├── params.py         # SimParams dataclass — all tunable parameters + from_physical classmethod
│   ├── particle.py       # Particle ABC with boundary logic; PassiveBrownianParticle; ActiveBrownianParticle
│   ├── simulator.py      # Simulator — time-steps a particle, returns trajectory array
│   ├── analysis.py       # orientation_acf, velocity_acf, position_acf
│   └── theory.py         # Analytical MSD + ACF formulas, incl. active_msd_exponent
├── visualization/
│   └── plotter.py        # plot_trajectory(), plot_msd(), plot_correlations()
├── tests/
│   ├── test_params.py    # Tests for SimParams boundary validation and from_physical
│   ├── test_particle.py  # Unit tests for all particle types and boundary modes
│   ├── test_simulator.py # Integration tests for Simulator (shape, determinism, MSD scaling)
│   ├── test_theory.py    # Unit tests for all analytical formulas in theory.py
│   └── test_analysis.py  # Tests for correlation function computation
├── docs/
│   ├── hPSC_parameters.md          # Literature review: hPSC radius, speed → calibrated D_T, D_R, v
│   ├── endoderm_migration_models.md # Review of DACF/MSD measurements and candidate model extensions
│   └── pdf_pages/                   # Source lecture-slide images (Sde Boker, Jan 2026) referenced above
├── main.py               # Entry point — saves 7 PNGs (trajectories, MSD+exponent, boundary comparisons, correlations, stop-vs-slip, theory comparison)
├── animate.py            # Live animation entry point
├── abp_cil.py            # ABP + steric repulsion + contact-inhibition-of-locomotion (N-cell ensemble)
├── active_fbm.py          # Active particle with fractional Gaussian translational noise (MSD ~ t^2H)
├── active_fbm_angular.py  # Active FBM with correlated (fGn) angular noise, shapes the DACF
├── hetero_abp.py          # Heterogeneous ABP: log-normal distribution of τ_R
├── hetero_abp_pareto.py   # Heterogeneous ABP: Pareto-distributed τ_R → exact power-law DACF
├── other images/          # Output plots from the exploratory scripts above
└── requirements.txt      # numpy, scipy, matplotlib, pytest
```

### File responsibilities in detail

| File | What it does |
|------|-------------|
| `simulation/params.py` | Single source of truth for all physics parameters. Change `D_T`, `D_R`, `v`, `dt`, `n_steps`, `seed`, `x0`, `y0`, `phi0` here. |
| `simulation/particle.py` | `Particle` is an abstract base class. Each subclass implements `step()` for one physics model. Adding a new model = add a new subclass. |
| `simulation/simulator.py` | `Simulator(particle, params).run()` returns a `(n_steps+1, 3)` NumPy array where columns are `[x, y, phi]`. Row 0 is the initial state. **Note:** `run()` mutates the particle in place — create a fresh particle for each run. |
| `simulation/theory.py` | Pure-function analytical predictions: `passive_msd`, `active_msd`, `active_msd_short_time`, `active_msd_long_time`, `active_msd_exponent`, `effective_diffusion`, `rotational_relaxation_time`. No simulation state. |
| `visualization/plotter.py` | `plot_trajectory(traj, title, ax)` draws a time-colored path. `plot_msd(trajs, dt, label, ax, theory_curves)` draws ensemble-averaged MSD on a log-log scale with optional theoretical overlays. |
| `main.py` | Demo script, calibrated to hPSC parameters (`docs/hPSC_parameters.md`). Saves 7 PNGs. Top of file has all constants (`D_T`, `D_R`, `V`, etc.) — edit there to experiment. Prints `tau_R`, `D_eff`, `Pe`, and persistence length on startup. |
| `animate.py` | Same constants at the top. `SKIP` controls playback speed (higher = faster). `TRAIL` controls how many past positions are shown. |
| `abp_cil.py` | N-cell ensemble: non-interacting ABP vs. steric repulsion vs. steric + contact-inhibition-of-locomotion (CIL). |
| `active_fbm.py` | Replaces translational white noise with fractional Gaussian noise (Hurst exponent H) to get MSD ~ t^2H. |
| `active_fbm_angular.py` | Adds correlated (fGn) angular noise on top of `active_fbm.py` to shape the directional ACF (DACF). |
| `hetero_abp.py` | Ensemble of standard ABPs with τ_R drawn from a log-normal distribution (cell-to-cell heterogeneity). |
| `hetero_abp_pareto.py` | Same idea with a Pareto-distributed τ_R, giving an exact power-law DACF. |

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
- Sde Boker Jan 2026 lecture notes — overall modeling context; slide images in `docs/pdf_pages/`, summarized in `docs/endoderm_migration_models.md`
- `docs/hPSC_parameters.md` — literature review behind the hPSC-calibrated parameters used in `main.py`
- Wadkin et al. 2017, *Sci. Reports* (PMC5428844); Hamada et al. 2018, *PLOS ONE* (PMC6130871); Hadjiantoniou et al. 2024, *Life* (PMC11595361) — hPSC migration speed measurements
