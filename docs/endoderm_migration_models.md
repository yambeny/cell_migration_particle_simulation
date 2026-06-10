# Endoderm Migration: Model Review and Recommendations

## Terminology

**DACF** — Directional AutoCorrelation Function: `⟨cos(φ(t+τ) − φ(t))⟩`
Correlates the *direction angle* only, independent of speed.
For the standard ABP this equals `exp(−D_R·τ)` — an exponential decay.
This is what we compute in our simulation as "Orientation ACF."

**VACF** — Velocity AutoCorrelation Function: `⟨v(t)·v(t+τ)⟩`
Correlates the full velocity vector (direction × speed).
For constant speed: VACF = v²·DACF.
For variable-speed cells they differ; the VACF carries extra information about speed fluctuations.

The Wiener–Khinchin relation connects VACF to MSD:
```
MSD(t) = 2 ∫₀ᵗ (t−s)·C_v(s) ds
```
- Exponential VACF (ABP/PRW): C_v ~ exp(−t/τ_R) → MSD → 4·D_eff·t at long times (α = 1, normal diffusion)
- Power-law VACF: C_v ~ t^{−β}              → MSD ~ t^{2−β} at all times   (α = 2−β, persistent anomalous diffusion)

---

## What the Sde Boker PDF Shows (Jan 2026 lecture)

**Key slides (pages 31–38):**

- **Page 34**: The DACF of endoderm cells is **not consistent with PRW** (persistent random walk)
  at any timescale, compared with 3T3 fibroblasts (which do follow PRW).
- **Page 35–36**: The model that reproduces endoderm 2D migration uses
  **power-law decaying directional memory**:
  ```
  θ_reference = Σᵢ wᵢ × θ_{t−i}     wᵢ = (i+1)^{−1.5}
  ```
- **Page 37**: Memory window ~1.5 hours.
- **Page 38**: *"Exponential models (PRW) don't fit all timescales; power-law memory does."*
  Summary: cells migrate slowly, exhibit power-law statistics.

**Measured MSD exponents (from the PDF context):**
- 3D: MSD ~ t^1.29
- 2D: MSD ~ t^1.4
- No visible crossover regime in either case.

---

## Why the Standard ABP Cannot Reproduce t^1.4

Our current model has DACF = exp(−D_R·τ), which is exactly a PRW.
The Wiener–Khinchin integral of an exponential VACF *always* returns to α=1 at t >> τ_R —
regardless of Pe, V, or D_R. This is not a quantitative failure (Pe too small);
it is a qualitative mismatch of model class.

Current hPSC parameters:
- Pe = v·R/D_T ≈ 0.74 → peak α ≈ 1.09 (from exact theory)
- Even at Pe = 10: peak α ≈ 1.6, but this returns to α=1 at long times

To produce a *persistent* MSD ~ t^1.4 without any crossover requires a power-law VACF,
not an exponential one.

---

## Alternative Models Considered

### 1. Standard ABP with Higher Pe
**Description:** Increase V or decrease D_R to push Pe >> 1.

**Pros:** Trivial to implement; larger intermediate super-diffusive window.

**Cons:** The crossover to α=1 is inevitable — it just shifts to longer times.
Cannot produce a true power-law over many decades. The PDF rules this out explicitly
(PRW fails at all timescales, not just long ones).

**Verdict: insufficient.**

---

### 2. Power-Law Directional Memory Model (from the PDF)
**Description:** Replace exponential angular diffusion with:
```
θ_ref = Σᵢ wᵢ θ_{t−i},   wᵢ = (i+1)^{−1.5}
θ_new = θ_ref + √(2D_R dt)·η
```
Memory window ~1.5 hours. Shown to reproduce 2D endoderm DACF and MSD.

**Pros:** Directly validated on the endoderm data; computationally simple.

**Cons (reasons for skepticism):**
- Purely **phenomenological** — no identified molecular mechanism for why
  past directions are weighted as a power law.
- The exponent α=1.5 is a fit parameter, not derived from first principles.
- The apparent single-cell memory could be an **emergent collective effect**:
  cells in a dense monolayer interact (CIL, adhesion), and if you track an individual
  cell within that collective, it appears to have long memory even though each cell
  individually has none. Applying the single-cell memory model to isolated cells
  would then be wrong.
- Cannot distinguish between intrinsic single-cell memory and collective
  interaction-induced pseudo-memory without single-cell isolation experiments.

**Verdict: fits the data but biologically unjustified for single cells.**

---

### 3. Fractional Brownian Motion (FBM, H = 0.7)
**Description:** Replace the position process with FBM: `x(t)` has
covariance `⟨x(t)x(s)⟩ = D·(|t|^{2H} + |s|^{2H} − |t−s|^{2H})`, H=0.7.
MSD ~ t^{2H} = t^{1.4} by construction.

**Pros:** Mathematically clean; H=0.7 directly reproduces t^1.4;
framework used by Dieterich et al. (PNAS 2008) for epithelial cells.

**Cons:** Also phenomenological — FBM describes long-range temporal correlations
but does not explain their biological origin. H is a fit parameter.
Simulation requires generating correlated Gaussian noise (Davies–Harte method, O(N log N)).

**Key reference:** Dieterich P et al., *PNAS* 2008 — anomalous dynamics of MDCK-F cells,
MSD ~ t^{1.25–1.28}.
[https://www.pnas.org/doi/10.1073/pnas.0707603105](https://www.pnas.org/doi/10.1073/pnas.0707603105)

**Verdict: clean and correct phenomenology, biologically agnostic.**

---

### 4. Heterogeneous ABP Population (Distributed τ_R)
**Description:** Each cell is a standard ABP, but τ_R is drawn from a
log-normal (or power-law) distribution across the population.
The ensemble-averaged MSD appears as a power law over the observed window.

**Pros:** Biologically very well motivated — cell-to-cell variability in cytoskeletal
state, adhesion strength, and morphology is real and well-documented.
Mechanistically justified as a population-level effect.
Reproduces apparent power law WITHOUT changing the single-cell model.

**Cons:** Individual cells remain purely diffusive at long times; the power law is
a population artifact. Requires knowing the distribution of τ_R, which must be
measured (DACF of individual cells, not ensemble).

**Key reference:** Campos D et al., *PLOS Comp Biol* 2019 — heterogeneous dynamics
as the correct mechanism for superdiffusion in mouse fibroblasts (α ~ 1.4–1.7).
[https://pmc.ncbi.nlm.nih.gov/articles/PMC6392322/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6392322/)

**Verdict: highly plausible for endoderm; testable by single-cell DACF fitting.**

---

### 5. Many-Body ABP with CIL (Contact Inhibition of Locomotion)
**Description:** Agent-based simulation where cells interact: upon contact,
the propulsion direction is repolarized away from the contact point
(Rac/Rho signaling). Creates effective anti-correlation in velocities after collisions.

**Pros:** Biologically grounded mechanism (CIL is well-established in endoderm;
see LaBelle et al. 2025). Could produce emergent long-range velocity correlations
that appear as power-law DACF at the single-cell level in a monolayer.
Naturally explains WHY the power-law memory model works phenomenologically.

**Cons:** Complex to implement correctly. Collective effect — harder to
interpret single-cell statistics. Requires calibrating interaction rules (CIL radius,
repolarization timescale).

**Key references:**
- LaBelle J et al., *iScience* 2025 — endodermal cells use CIL during migration.
  [https://www.cell.com/iscience/fulltext/S2589-0042(25)02405-8](https://www.cell.com/iscience/fulltext/S2589-0042(25)02405-8)
- Basan M et al., *PLOS Comp Biol* 2021 — neighbor-enhanced diffusivity in
  adhesive+active cell monolayers raises α from 1.07 to 1.24.
  [https://pmc.ncbi.nlm.nih.gov/articles/PMC8491951/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8491951/)

**Verdict: most physically complete; best long-term model; significant implementation effort.**

---

### 6. Run-and-Tumble with Lévy-Distributed Run Times
**Description:** Cells propel for a duration τ drawn from P(τ) ~ τ^{−(1+μ)},
1 < μ < 2. Ensemble MSD ~ t^{3−μ}. For α=1.4: μ=1.6.

**Pros:** Produces exact power law with no crossover. Power-law run-time
distributions have been reported for T-cells and dendritic cells.

**Cons:** No direct evidence for power-law run times in endoderm specifically.
Requires detailed analysis of run-time distributions from tracking data.

**Verdict: possible but unvalidated for endoderm.**

---

## Measured Parameters from Sde Boker PDF (Jan 2026)

Extracted from slide images (pages 21, 32–34, 36–37) — most data is in figures.

| Quantity | Micropattern (3D-like) | 2D | Notes |
|----------|----------------------|-----|-------|
| MSD fit | 125.44·t^**1.29** | 193.92·t^**1.40** | t in hours, MSD in µm² |
| DACF decay | power law ~t^{−0.78} | power law ~t^{−0.83} | NOT exponential |
| Nucleus area (median) | 73.3 µm² → R_nuc ≈ 4.8 µm | 265.8 µm² (cells spread in 2D) | cell R ≈ 6–7 µm |
| Cell speed | ~20–40 µm/h (log-normal hist.) | similar | mean ≈ **30 µm/h = 0.0083 µm/s** |
| Inward bias (chemotaxis) | mean −2.08 µm/h (504 tracks) | — | 68% of cells migrate inward |
| Memory window | ~1.5–3 h | ~1.5–3 h | power-law model fit |
| Effective τ_R (biological) | **~90 min = 5400 s** | similar | from memory window; not Stokes-Einstein |

**Critical observation (page 34):** 3T3 fibroblasts show exponential DACF (PRW-consistent);
both endoderm conditions show clean power-law DACF — the cells genuinely violate ABP/PRW
at the single-cell level.

---

## Why Biological Parameters Change Everything: Pe Analysis

The problem with our earlier simulation was using **Stokes-Einstein D_R** (thermal rotation).
For living cells, persistence is set by cytoskeletal remodelling, not thermal fluctuations.

| Parameter | Stokes-Einstein | Biological (PDF) |
|-----------|----------------|-----------------|
| v | 0.004 µm/s (= 14.4 µm/h) | **0.0083 µm/s (= 30 µm/h)** |
| D_R | 6.20×10⁻⁴ rad²/s | **1.85×10⁻⁴ rad²/s** (= 1/5400 s) |
| τ_R | 27 min | **90 min** |
| τ_T = 4D_T/v² | 145 min (> τ_R) | **34 min (< τ_R)** |
| Pe = v²/(4D_T D_R) | 0.80 | **2.68** |
| Ballistic regime? | No (τ_T > τ_R) | **Yes (τ_T < τ_R)** |

The single flip from thermal to biological D_R (a factor of 3.3×) changes Pe from 0.8 → 2.7,
which switches the system from "no ballistic regime" to "clear ballistic regime."

### Resulting MSD exponents (computed numerically)

**Single ABP, biological parameters (v = 30 µm/h, τ_R = 90 min):**
- Peak local α = **1.48** at t ≈ 76 min ≈ 0.85 τ_R
- Global power-law fit over observation window [0.1, 10 h]: α = **1.385**

**Heterogeneous ABP (log-normal τ_R, mean = 90 min), biological v:**

| σ_log (spread of ln τ_R) | τ_R 5th–95th percentile | Global α fit |
|--------------------------|------------------------|-------------|
| 0.5 | 42–203 min | **1.389** |
| 1.0 | 19–519 min | **1.401** ← matches 2D data |
| 1.5 | 9–1202 min | **1.423** |

**Conclusion:** Using the PDF biological parameters, a single ABP already gives α ≈ 1.39.
A modest population spread (σ_log ≈ 1.0) gives α = 1.40 — exactly the measured 2D value.
Heterogeneous ABP is viable here because Pe > 1, unlike with Stokes-Einstein parameters.

---

## Summary and Recommendation

| Model | Gets t^1.4? | Biological basis | Effort |
|-------|------------|-----------------|--------|
| ABP higher Pe | No (transient) | Low | Trivial |
| Power-law memory | Yes | Low (phenomenological) | Low |
| FBM H=0.7 | Yes (by construction) | Low (phenomenological) | Low–Medium |
| Heterogeneous ABP | Apparent | High (population variability) | Medium |
| Many-body ABP + CIL | Possibly emergent | High (CIL well-established) | High |
| Lévy run times | Yes | Medium | Medium |

**Recommended path:**

1. **First**: Measure individual-cell DACF from tracking data.
   - If each cell's DACF is exponential with variable τ_R → heterogeneous ABP is correct.
   - If each cell's DACF is non-exponential (power-law from the start) → intrinsic single-cell mechanism (FBM or Lévy).

2. **For simulation now**: Implement heterogeneous ABP (log-normal τ_R distribution)
   as the most biologically plausible model that can be built directly on the current code.

3. **Longer term**: Many-body ABP + CIL to model collective behavior and tube formation.

---

## On the Directional Memory Model

Skepticism is well-founded. The model works numerically but:
- No molecular mechanism connects (i+1)^{−1.5} to any known biology.
- The 1.5-hour memory window could be collective, not intrinsic.
- It is a black-box curve-fit, not a predictive physical model.
- It cannot be used to predict what happens under perturbations
  (drug treatment, substrate change, density change) without re-fitting.

A model with biological motivation (CIL, variable τ_R) is preferable even if it
requires more effort, because it connects to measurable molecular quantities.

---

---

## Heterogeneous ABP: Implementation Plan

### Concept
Each simulated cell is a standard ABP, but τ_R is drawn independently from a log-normal
distribution reflecting real cell-to-cell variability in cytoskeletal persistence.
No new physics required — only a parameter distribution on top of existing code.

### Parameters to use (from PDF biological values)
```python
RADIUS_UM     = 6.5       # µm  (cell radius; nucleus area 73.3 µm² → R_nuc≈4.8 µm)
V             = 0.0083    # µm/s  (= 30 µm/h from PDF velocity histogram)
T_K           = 310.0     # K
ETA_PA_S      = 1e-3      # Pa·s
# D_T from Stokes-Einstein (thermal, unchanged)
# D_R: set BIOLOGICALLY, NOT from Stokes-Einstein:
TAU_R_MEAN    = 5400.0    # s  (= 90 min from PDF memory window)
TAU_R_SIGMA   = 1.0       # spread of ln(τ_R); 1.0 covers ~19–519 min (5th–95th percentile)
```

### Changes to code
1. **`main.py`**: replace single `N_ENSEMBLE` runs with a loop that draws `τ_R_i` per cell:
```python
import numpy as np

rng = np.random.default_rng(seed=0)
tau_R_samples = np.exp(rng.normal(np.log(TAU_R_MEAN), TAU_R_SIGMA, N_ENSEMBLE))

heterogeneous_trajs = [
    _run(ActiveBrownianParticle,
         SimParams(D_T=D_T, D_R=1.0/tau_R_i, v=V, dt=DT_SIM, n_steps=N_STEPS_MSD, seed=s))
    for s, tau_R_i in enumerate(tau_R_samples)
]
```

2. **`DT_SIM` and `N_STEPS_MSD`**: set to cover the observation window [0.1, 10 h]:
```python
DT_SIM      = 60.0   # s  (1 min steps)
N_STEPS_MSD = 600    # 600 × 60 s = 36000 s = 10 h
```

3. **Plot**: log-log MSD comparing homogeneous ABP vs heterogeneous ABP vs power-law fit t^1.4.
   Overlay the global power-law fit to extract α and compare to measured 1.29 (micropattern)
   and 1.40 (2D).

4. **Validation test**: plot the distribution of τ_R values drawn, and verify that the
   ensemble DACF (orientation ACF) is non-exponential when averaged over heterogeneous cells.

### Expected outcome
- Single ABP with biological v and τ_R: global α ≈ 1.39 (close to 2D measurement)
- Heterogeneous ABP (σ_log = 1.0): global α ≈ 1.40 (matches 2D exactly)
- The DACF of the ensemble will appear as a power law (curved on log-log) rather than
  a single exponential — consistent with the PDF observation

### Known limitations
- Individual cells still have exponential DACF; the ensemble power-law is a population artefact
- If single-cell trajectories are long enough to see the exponential cutoff, this model
  will be distinguishable from true FBM or power-law memory
- The biological D_R is not predicted from first principles — it must be fitted from DACF data

---

---

## Model Progression: From Single-Cell to Collective

### Why heterogeneous ABP is not the end of the story

The heterogeneous ABP (`hetero_abp.py`) reproduces the correct MSD *exponent* (α ≈ 1.40) using
biological parameters from the PDF, but has three structural failures:

1. **DACF shape wrong**: each cell still has an exponential DACF; only the ensemble average
   looks pseudo-power-law over the 10 h observation window. The PDF shows power-law DACF on
   *individual* cells (or at least on ensemble measurements that cannot be explained by
   heterogeneity alone).
2. **Amplitude wrong**: MSD amplitude is ~6× too high at t=1 h. The effective swim diffusion
   `D_eff = v²/(2D_R)` with v=30 µm/h and τ_R=90 min gives `D_eff_swim = 675 µm²/h`, but the
   measured MSD at 1 h is only ~194 µm². To fix the amplitude, v and τ_R must be *jointly fitted*
   to the MSD data, not read from separate figures.
3. **No mechanistic prediction**: v and D_R are free parameters with no predictive power for
   perturbations (drug, substrate, density).

---

### The key diagnostic experiment

**Single-cell isolation**: track endoderm cells as isolated individuals (not in a monolayer).

| Outcome | Interpretation | Correct model |
|---------|---------------|---------------|
| Isolated cell DACF is exponential, monolayer DACF is power-law | Power law is emergent/collective | ABP + CIL / many-body |
| Isolated cell DACF is power-law | Power law is intrinsic single-cell | FBM / GLE / Active FBM |

All modelling decisions below depend on which scenario is true. The PDF does not report isolation
experiments, so both remain plausible.

---

### Path A: Intrinsic single-cell mechanisms

These replace the ABP equation of motion with one that naturally produces power-law DACF.
No agents or cell-cell interactions required.

**Active FBM (Fractional Brownian Motion)**
Replace angular white noise with fractional Gaussian noise (Hurst exponent H):
```
dφ/dt = ξ_H(t)     with  <ξ_H(t) ξ_H(s)> ~ |t-s|^{2H-2}
```
- H=0.5: standard ABP (exponential DACF). H>0.5: persistent long-range correlations.
- DACF ~ t^{-(2-2H)}, MSD ~ t^{1+H}. For α=1.4: H≈0.7, DACF ~ t^{-0.6}.
- Biologically unjustified *per se* but connects to the GLE model below.

**Generalised Langevin Equation (GLE) — viscoelastic ECM**
Replace simple Stokes drag with a memory kernel:
```
γ(t) ~ t^{-(1-β)}   →   MSD ~ t^{1+β},  DACF ~ t^{-β}
```
The ECM is viscoelastic (measured rheology of collagen/fibronectin substrates gives β ≈ 0.3–0.5).
This is the most **physically motivated** single-cell model: the power law arises from the
substrate mechanics, not from the cell's internal state. Self-propulsion can be layered on top.
Equivalent to active FBM at the stochastic level.
Key reference: Kou & Xie, PRL 2004; Weber et al., PNAS 2012 (in vivo FBM of chromosomal loci).

---

### Path B: Collective/emergent mechanisms

These keep each cell as a standard ABP but add cell-cell interactions that create emergent
anomalous diffusion at the population level.

**ABP + CIL (implemented in `abp_cil.py`)**
When two cells contact (|r_ij| < 2R), cell i receives an angular torque repolarising φ_i
away from the contact direction:
```
dφ_i/dt = sqrt(2 D_R) η + k_CIL * sum_{j in contact} sin(θ_away_ij − φ_i)
```
CIL creates anti-correlations at contact timescale and (in a dense monolayer) produces
emergent long-range velocity correlations at the population level. Each individual cell
still has exponential DACF; the effective DACF of a tracked cell in the monolayer
appears slower due to correlated re-encounters.

Physical mechanism: Rac/Rho signalling at contact is well-established in endoderm
(LaBelle et al. 2025). The anti-persistence after collision makes a cell "remember" its
path for longer than τ_R.

Parameters used in `abp_cil.py`:
```python
N_CELLS   = 100           # interacting cells in periodic box
PACK_FRAC = 0.60          # area packing fraction (endoderm monolayer)
K_CIL     = 1.0 / 600.0  # s^{-1}: CIL rate (tau_CIL = 10 min; faster than τ_R=90 min)
K_REP     = 0.02          # s^{-1}: soft steric repulsion (prevents overlap)
D_CONTACT = 13.0          # µm: contact threshold = 2R
DT_SIM    = 30.0          # s (smaller than hetero_abp.py for steric stability)
```

Three conditions are simulated to isolate effects:
1. ABP, no interaction (K_REP=0, K_CIL=0) — reference, should match ABP theory
2. ABP + steric only (K_REP>0, K_CIL=0) — crowding effect alone
3. ABP + steric + CIL (K_REP>0, K_CIL>0) — full model

Expected behaviour:
- Crowding alone: slightly reduces α (cells block each other, reducing effective D_eff)
- CIL on top: repolarisation away from contacts → net increase in directional persistence
  between contacts → α increases relative to steric-only

Basan et al. PLOS CB 2021 found adhesion + ABP raises α from 1.07 → 1.24 in dense monolayers.
CIL with biological Pe > 1 is expected to give a stronger effect, potentially reaching 1.4.

**Active Vertex Model** (not yet implemented)
Cells are deformable polygons with line tension, cell pressure, motility. Naturally captures
jamming transition, cell shape index, tissue fluidity. Appropriate for tube formation and
large-scale tissue mechanics but significant implementation overhead.

---

---

## ABP+CIL Results (abp_cil.py)

Three conditions at φ=0.60, N=100, biological params:

| Condition | global α | vs ABP theory |
|-----------|---------|--------------|
| ABP, no interaction (K_REP=0, K_CIL=0) | 1.35 | −0.01 (noise only) |
| ABP + steric only | **1.19** | **−0.16** (crowding reduces α!) |
| ABP + steric + CIL | **1.19** | −0.16 + 0.003 (CIL negligible at this density) |

**Key finding**: crowding at φ=0.60 REDUCES α by −0.16 (caging effect). The steric forces
act as additional random kicks, effectively increasing angular randomisation and shortening the
effective τ_R. CIL on top adds essentially nothing (+0.003): at φ=0.60 each cell simultaneously
contacts 3–5 neighbours in different directions, so CIL torques cancel out.

**Physical interpretation**: CIL is most effective when contacts are isolated (low density).
In a dense monolayer, the crowding/jamming effect dominates and suppresses superdiffusion.
The measured α=1.40 in the PDF monolayer is therefore NOT explained by ABP+CIL at physiological
packing — either the measurement is at lower density, or there is additional active coordination
(polar alignment / chemotaxis) that we have not modelled, or the intrinsic single-cell mechanism
(FBM/GLE) is what actually produces α=1.40.

**On Rac/Rho and polar alignment**: Rac and RhoA are small GTPases that control the actin
cytoskeleton. Rac at the leading edge drives protrusion; Rho at the rear drives retraction.
They are mutually inhibitory, creating cell polarity. CIL is their collision consequence
(Rho activated at contact → repolarise away). Polar alignment would instead mean cells match
their propulsion direction to their neighbours — implementable by replacing `sin(θ_away_ij − φ_i)`
with `sin(φ_j − φ_i)` — but requires a different biological justification (cadherin tension
transmission, chemokine alignment) and was not the dominant effect here.

---

## Active FBM Results (active_fbm.py)

**What is FBM?**

Standard Brownian motion has *independent* increments: each step is uncorrelated with all
previous steps. Fractional Brownian Motion (FBM) generalises this by introducing *long-range
temporal correlations* between increments, controlled by the Hurst exponent H ∈ (0,1):

- H = 0.5 : standard BM (no memory, normal diffusion, MSD ~ t^1)
- H > 0.5 : *persistent* — a step in one direction makes the next step more likely in the
             same direction. Positive correlation decays as a power law, never fully dying out.
             MSD ~ t^{2H} (superdiffusion, α > 1).
- H < 0.5 : *anti-persistent* — a step is more likely followed by a reversal. Subdiffusion.

For endoderm: MSD ~ t^{1.4} → 2H = 1.4 → H = 0.7.

**Physical motivation (GLE)**: a cell on a viscoelastic ECM experiences a memory friction:
```
γ(t) ~ t^{-(1-β)}    (power-law stress relaxation of the ECM)
```
This is equivalent in the overdamped limit to fBM translational noise with H = (1+β)/2.
Measured collagen/fibronectin gels have β ≈ 0.3–0.4, giving H ≈ 0.65–0.70. No free parameter!

**Active FBM model (active_fbm.py)**:
Standard self-propulsion + fGn translational noise:
```
x(t+dt) = x(t) + v*cos(phi)*dt + xi_H(t)
phi(t+dt) = phi(t) + sqrt(2*D_R*dt)*eta          [angular: unchanged]
```
Analytical MSD (exact):
```
MSD(t) = 2*D_H*t^{2H}  +  (v^2/D_R)*(2t - 2*tau_R*(1-exp(-t/tau_R)))
          fBM term              active swim term (from ABP)
```

Normalisation: D_H = D_T * dt^{1-2H} so the per-step noise amplitude is the same for all H.
Only the *correlations* between steps differ.

| H | D_H | global α (10 h window) | long-time limit | vs 1.40 |
|---|-----|----------------------|-----------------|---------|
| 0.5 | D_T | 1.40 | 1.0 | ±0.00 |
| 0.6 | 0.44·D_T | 1.36 | 1.2 | −0.04 |
| 0.7 | 0.19·D_T | 1.46 | 1.4 | +0.06 |
| 0.8 | 0.09·D_T | 1.65 | 1.6 | +0.25 |

**Key finding**: H=0.5 (standard ABP with biological parameters) ALREADY gives α=1.40 in the
10 h observation window — because the window sits inside the biological ballistic regime.
H=0.7 gives α=1.46 (slightly too high) because the fBM term adds extra superdiffusion on
top of the already-present ballistic regime.

**What Active FBM does NOT fix**: the DACF remains exponential (angular process unchanged).
All H values give DACF ~ exp(−t/τ_R). The power-law DACF in the PDF requires *correlated angular
noise*, not correlated translational noise — this is a separate model (GLE for rotation, or
the power-law memory model from the PDF).

**Conclusion**: Active FBM adds a physically motivated mechanism (viscoelastic ECM → power-law
substrate memory) but does not produce a better MSD exponent than the standard biological ABP
over the 10 h window. Its value is in the *long-time prediction*: standard ABP returns to α=1
after ~10·τ_R (~15 h), while Active FBM with H=0.7 maintains α=1.4 forever. These are
experimentally distinguishable only with tracks > 15 h.

---

## Active FBM with Correlated Angular Noise (active_fbm_angular.py)

**Motivation**: every model so far leaves the angular process as white noise, guaranteeing an
exponential DACF. The PDF shows a power-law DACF ~ t^{-0.83}. To reproduce this, the orientation
increments themselves must have long-range temporal structure.

**Key insight — which direction of H_phi matters**:

For angular fBM with Hurst exponent H_phi and per-step variance equal to standard white noise:
```
Var(phi(t+tau) - phi(t)) = 2*D_R * tau^{2*H_phi} * dt^{1-2*H_phi}
DACF(tau) = exp(-D_R * tau^{2*H_phi} * dt^{1-2*H_phi})
```

- H_phi = 0.5 : standard exponential DACF (baseline)
- H_phi > 0.5 : persistent rotation — each turn continues in the same direction — phi
                diffuses FASTER, DACF decays FASTER than exponential. **Wrong direction.**
- H_phi < 0.5 : antipersistent rotation — each turn is slightly reversed — phi
                accumulates angle MORE SLOWLY, DACF decays SLOWER than exponential. **Correct direction.**

The PDF DACF ~ t^{-0.83} decays *slower* than the standard exponential (whose log-log slope
steepens to ~ −1.3 over 0.5–5 h). Therefore we need **H_phi < 0.5**.

**Physical interpretation**: antipersistent angular increments arise naturally when the
cytoskeleton "overshoots" in turning — a rightward rotation is corrected by a leftward one at
the next timescale, creating a zigzag at short times but slower net orientation loss at long times.
This is consistent with the viscoelastic GLE picture for the rotational degree of freedom: the
same ECM memory that slows translational diffusion also damps rotational diffusion.

**Calibration**: to target DACF log-log slope ≈ −0.83 over the 0.5–5 h window:
```
2*H_phi ≈ 0.92   →   H_phi ≈ 0.46
```
(Analytical estimate confirmed numerically.)

**Model equations**:
```
x(t+dt)   = x(t)   + v*cos(phi)*dt  +  xi_H_trans(t)   [fGn, H_trans; scale=sqrt(2*D_T*dt)]
y(t+dt)   = y(t)   + v*sin(phi)*dt  +  xi_H_trans(t)
phi(t+dt) = phi(t) + zeta_H_phi(t)                      [fGn, H_phi;   scale=sqrt(2*D_R*dt)]
```
Both fGn sequences use the same per-step variance as white noise. All memory is encoded in the
Cholesky covariance structure.

**Simulation results** (N=300, 10 h, biological parameters):

| H_trans | H_phi | alpha (MSD) | DACF slope | Notes |
|---------|-------|-------------|------------|-------|
| 0.5 | 0.50 | 1.34 | −1.42 | Standard ABP reference |
| 0.7 | 0.50 | 1.45 | −1.61 | Trans. fBM only — DACF worse |
| 0.5 | 0.46 | 1.44 | −0.93 | Ang. fBM only — DACF close |
| **0.7** | **0.46** | **1.49** | **−0.88** | **Full model — both close** |
| 0.7 | 0.48 | 1.46 | −0.99 | DACF too steep |

**Targets**: α = 1.40, DACF slope = −0.83.

**What the DACF shape actually is — and why this does NOT recover the power law**:

The fGn angular process gives a *stretched exponential* `exp(-c·tau^{2H_phi})`, never a true
power law. The instantaneous log-log slope of a stretched exponential is NOT constant:

```
d(ln DACF)/d(ln tau) = -D_R * 2*H_phi * tau^{2*H_phi} * dt^{1-2*H_phi}
```

For H_phi=0.46, this slope ranges from ≈ −0.24 at tau=0.5 h to ≈ −1.94 at tau=5 h.
The fitted value of −0.88 is merely a linear average over this curved line — the **shape is
still qualitatively wrong**: concave-upward on log-log, not a straight line.

A power law `τ^{-0.83}` is a straight line on log-log with constant slope −0.83 at every lag.
The stretched exponential only passes through the "correct" slope at one intermediate time
(around 2 h here), then veers away in both directions.

**Conclusion**: fGn angular noise with H_phi < 0.5 is NOT sufficient to reproduce the DACF.
It shifts the average slope towards the target but preserves the exponential-family shape.
The fitted "DACF slope" metric is misleading here — it hides the shape mismatch.

**Remaining limitations**:
- MSD amplitude still ~6× too high (same as all models: v and τ_R need joint fitting to the MSD
  curve, not independent extraction from figures)
- The DACF is technically a stretched exponential, not a true power law
- H_trans and H_phi are fit parameters; physical motivation exists (GLE/viscoelastic ECM) but
  the specific values (0.7, 0.46) have not been derived from ECM rheology measurements

---

### Model comparison summary (updated)

| Model | Code file | α | DACF slope | Mechanistic? | Notes |
|-------|-----------|---|------------|-------------|-------|
| Standard ABP (Stokes-Einstein) | `main.py` | 1.09 ✗ | −1.4 ✗ | No | Pe<1, no ballistic regime |
| ABP (biological D_R) | `main.py` | 1.37 ~✓ | −1.4 ✗ | Partial | Pe=2.67, τ_R=90 min |
| Heterogeneous ABP (σ=1.0) | `hetero_abp.py` | 1.40 ✓ | −1.4 ✗ (pseudo) | Partial | Pop. spread; individual cells still exponential |
| ABP + CIL at φ=0.60 | `abp_cil.py` | 1.19 ✗ | −1.4 ✗ | Yes | Crowding suppresses α; CIL cancels at high density |
| Active FBM (H_trans=0.7) | `active_fbm.py` | 1.46 ~✓ | −1.6 ✗ | Yes (GLE) | Translational memory; DACF worsens |
| Active FBM + angular (H_t=0.7, H_phi=0.46) | `active_fbm_angular.py` | 1.49 ~✓ | −0.88 ✗ (avg slope only; shape still curved) | Yes (GLE) | Stretched exponential, not power law |
| Active Vertex Model | not yet | TBD | TBD | Yes | Needed for tube formation / jamming |

**Targets**: α = 1.40, DACF slope = −0.83.

**No model up to active_fbm_angular.py produces a true power-law DACF.**  All produce
exponential-family DACF (pure exponential, mixture of exponentials, or stretched exponential).
See the Pareto heterogeneous ABP section below for the model that resolves this.

---

## Pareto Heterogeneous ABP (hetero_abp_pareto.py)

**Why Pareto and not log-normal?**

The ensemble DACF of any heterogeneous ABP population is a Laplace transform of P(τ_R):
```
DACF(τ) = <exp(-τ/τ_R)> = ∫ P(τ_R) exp(-τ/τ_R) dτ_R
```
For a log-normal P(τ_R), this integral gives a stretched exponential — never a true power law.
For a Pareto P(τ_R) ~ τ_R^{-(1+β)}, the Tauberian theorem gives:
```
DACF(τ)  →  Γ(1+β) · τ_min^β · τ^{-β}    for τ >> τ_min
```
This is an **exact power law** with exponent β — a straight line on log-log.

**Physical motivation**: A power-law distribution of τ_R means that while most cells are
moderately persistent (~τ_min), there is no upper bound on how persistent a cell can be —
a fraction ~ (τ_min/τ_R)^β of cells have persistence time ≥ τ_R. Biologically, this reflects
a scale-free distribution of cytoskeletal states: actin network organisation has no preferred
timescale, and occasionally produces cells that stay polarised for hours.

**Parameters** (tau_min=2400s = 40 min, tuned so median τ_R = 91 min ≈ biological 90 min):
```
P(τ_R) = β · τ_min^β · τ_R^{-(1+β)},   τ_R ≥ τ_min
τ_min = 2400 s = 40 min   (minimum persistence time; power law holds for τ >> τ_min = 0.67 h)
β     = 0.83              (sets DACF exponent; tuned directly to measured value)
```

**Results** (N=800, tau_min=2400s):

| β | median τ_R | alpha (MSD) | DACF shape | vs targets (1.40, −0.83) |
|---|---|---|---|---|
| 0.50 | 154 min | 1.51 | power law t^{−0.50} | alpha high, DACF wrong exponent |
| 0.70 | 112 min | 1.45 | power law t^{−0.70} | alpha ok, DACF slope wrong |
| **0.83** | **91 min** | **1.42** | **power law t^{−0.83}** | **both match** |
| 1.00 | 82 min | 1.39 | power law t^{−1.00} | alpha ok, DACF too steep |
| Log-normal σ=1.0 | 90 min | 1.41 | stretched exponential | alpha ok, DACF shape wrong |

**Key finding**: the Pareto model produces an analytically exact power-law DACF ~ τ^{-β} for
τ >> τ_min. BUT there is a fundamental trade-off between the two observables:

```
tau_min small (≤600 s, 10 min):  power law visible from 0.5 h  ✓  |  alpha ≈ 1.24  ✗
tau_min large (=2400 s, 40 min): alpha ≈ 1.42                  ✓  |  power law only starts at 2–3 h  ✗
```

With τ_min=2400s, the local log-log slope of the DACF is:
- τ = 1 h: −0.39 (far from −0.83)
- τ = 5 h: −0.82 (close, but barely one point)

The *entire* 0.5–5 h observation window is inside the transition from 1 → power law. There is
no visible straight line on the log-log plot.

**Conclusion**: no pure Pareto parameterisation simultaneously gives a visually clear power-law
DACF over the 0.5–5 h window AND alpha ≈ 1.40. These two requirements pull τ_min in opposite
directions and cannot both be satisfied at once.

**What this implies**: the power-law DACF in the data is most likely NOT a population-heterogeneity
effect (if it were, a Pareto distribution would work with the right τ_min). It is more likely
intrinsic to each cell — meaning even isolated cells would show a power-law DACF. This points
to the single-cell GLE model: power-law memory in the angular process itself, arising from
viscoelastic cytoskeletal dynamics. The diagnostic experiment (track isolated cells) is the
only way to confirm this.

---

### Final model comparison (all models)

| Model | Code | alpha | DACF | Notes |
|-------|------|-------|------|-------|
| Standard ABP (Stokes-Einstein) | `main.py` | 1.09 ✗ | exponential ✗ | Pe<1 |
| ABP (biological D_R) | `main.py` | 1.37 ~✓ | exponential ✗ | Pe=2.67 |
| Heterogeneous ABP, log-normal σ=1.0 | `hetero_abp.py` | 1.40 ✓ | stretched exp. ✗ | shape wrong |
| ABP + CIL at φ=0.60 | `abp_cil.py` | 1.19 ✗ | exponential ✗ | crowding suppresses α |
| Active FBM H=0.7 (translational) | `active_fbm.py` | 1.46 ~✓ | exponential ✗ | DACF unchanged |
| Active FBM + angular H_phi=0.46 | `active_fbm_angular.py` | 1.49 ~✓ | stretched exp. ✗ | avg slope ok, shape wrong |
| Pareto hetero ABP β=0.83, τ_min=2400s | `hetero_abp_pareto.py` | 1.42 ✓ | no straight line in window ✗ | τ_min too large for window |
| Pareto hetero ABP β=0.83, τ_min=600s | `hetero_abp_pareto.py` | 1.24 ✗ | power law from 1h ✓ | alpha too low |

**Conclusion**: no implemented model simultaneously reproduces both observables. The Pareto
model proves that population heterogeneity CAN produce a power-law DACF analytically, but
the required τ_min conflicts with the alpha requirement. The most likely interpretation is
that the power-law DACF is intrinsic to single cells, not a population artifact — requiring
either a single-cell GLE angular model or the isolation experiment to confirm.

---

## Key References

| Paper | Finding | Relevance |
|-------|---------|-----------|
| Dieterich et al., PNAS 2008 | MSD ~ t^{1.25–1.28} in epithelial cells; FBM framework | Direct analogy to endoderm |
| Campos et al., APL Bioengg 2019 ([PMC6324209](https://pmc.ncbi.nlm.nih.gov/articles/PMC6324209/)) | Anomalous diffusion in 2D/3D hydrogels; α=1.4–1.8 | Parameter range |
| Campos et al., PLOS Comp Biol 2019 ([PMC6392322](https://pmc.ncbi.nlm.nih.gov/articles/PMC6392322/)) | Heterogeneous τ_R distribution explains ensemble α=1.4–1.7 | Key mechanism |
| Basan et al., PLOS Comp Biol 2021 ([PMC8491951](https://pmc.ncbi.nlm.nih.gov/articles/PMC8491951/)) | Adhesion raises α from 1.07 to 1.24 in dense monolayers | Cell-cell interaction effect |
| LaBelle et al., iScience 2025 | CIL in endodermal cell migration | Biological mechanism |
| Furusawa et al., PLOS ONE 2018 ([PMC6130871](https://pmc.ncbi.nlm.nih.gov/articles/PMC6130871/)) | hiPSC-derived mesendoderm migration random walk | hPSC baseline |
| Szamel et al., EPJ E 2021 | MCT for dense ABP; D_R^eff reduced by crowding | Many-body mean field |
| Romanczuk et al., EPJ Spec Top 2012 | ABP review: MSD regimes, Pe dependence | Theory backbone |
| Sde Boker Jan 2026 (lecture PDF) | Power-law DACF; memory window 1.5 h; PRW fails | Direct experimental source |
