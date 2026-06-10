"""Heterogeneous ABP with Pareto-distributed persistence times.

The log-normal distribution in hetero_abp.py produces a mixture of exponentials
that only APPROXIMATES a power-law DACF over a limited window.  A Pareto
(power-law) distribution of tau_R produces an EXACT power-law DACF:

    P(tau_R) = beta * tau_min^beta / tau_R^{1+beta},   tau_R >= tau_min

    DACF(tau) = <exp(-tau/tau_R)>
              = Gamma(1+beta) * tau_min^beta * tau^{-beta}   [tau >> tau_min]

This follows from the Laplace transform of the Pareto distribution (Tauberian theorem).
The power law holds from tau ~ tau_min onwards; at shorter lags DACF -> 1.

Physical interpretation
-----------------------
A heavy-tailed distribution of persistence times is natural if cells differ in their
cytoskeletal state.  Cells with long tau_R are highly polarised (strong Rac activity,
aligned actin fibres); cells with short tau_R are unpolarised.  If a small fraction of
cells are extremely persistent (Pareto tail), they dominate the long-time DACF and
pull the ensemble average towards a power law.

Why Pareto and not log-normal?
- Log-normal: DACF decays as a stretched exponential.  Shape is always concave-up on
  log-log (slope varies from shallow to steep).  Only matches power law on average.
- Pareto: DACF is an exact power law ~ tau^{-beta} for tau >> tau_min.  Shape is a
  straight line on log-log.  Exponent beta is directly the shape parameter.

Key parameters
--------------
beta   = DACF power-law exponent (target: 0.83 from PDF 2D data)
tau_min = minimum persistence time.  Power law holds for tau >> tau_min.
          Here: tau_min = 600 s (10 min) so the power law covers 0.5-5 h.

Reference: docs/endoderm_migration_models.md
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import gamma as gamma_func

from simulation.params import SimParams
from simulation.particle import ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf
from simulation.theory import active_msd

_HERE = Path(__file__).parent

# ── Physical parameters (same as hetero_abp.py) ───────────────────────────────
RADIUS_UM  = 6.5
T_K        = 310.0
ETA_PA_S   = 1e-3
V_BIO      = 0.0083    # µm/s = 30 µm/h
TAU_R_MEAN = 5400.0    # s  = 90 min (used as reference / log-normal mean)

_phys   = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
D_T     = _phys.D_T

# ── Simulation parameters ──────────────────────────────────────────────────────
DT_SIM     = 60.0
N_STEPS    = 600    # 10 h
N_ENSEMBLE = 800    # large ensemble: Pareto tail needs good sampling

# ── Pareto parameters ──────────────────────────────────────────────────────────
# tau_min: power law holds for tau >> tau_min. Use 600 s so law covers 0.5-5 h.
# tau_min controls where the power law onset is AND the typical cell persistence.
# Larger tau_min -> more persistent typical cell -> higher alpha.
# Power law holds for tau >> tau_min; needs tau_min << 0.5 h = 1800 s.
# tau_min = 1500 s (25 min): median tau_R = 1500 * 2^{1/0.83} = 1500 * 2.31 = 58 min
# tau_min = 2400 s (40 min): median tau_R = 2400 * 2^{1/0.83} = 2400 * 2.31 = 92 min
# This matches the biological tau_R_mean = 90 min at the MEDIAN of the Pareto distribution.
TAU_MIN    = 2400.0  # s = 40 min

# beta values to explore (DACF ~ tau^{-beta})
BETA_VALUES = [0.5, 0.7, 0.83, 1.0]
COLORS      = {0.5: "tab:green", 0.7: "tab:orange", 0.83: "tab:red", 1.0: "tab:purple"}


def sample_pareto(beta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from Pareto(tau_min, beta): P(x) ~ x^{-(1+beta)}, x >= tau_min.

    numpy.Generator.pareto(a) draws from the standard Pareto with minimum 1 and shape a.
    Shift and scale: tau_R = tau_min * (1 + rng.pareto(beta, n))
    """
    return TAU_MIN * (1.0 + rng.pareto(beta, n))


def dacf_pareto_theory(tau: np.ndarray, beta: float) -> np.ndarray:
    """Analytical DACF for Pareto-distributed tau_R.

    DACF(tau) = Gamma(1+beta) * tau_min^beta * tau^{-beta}   for tau >> tau_min.
    Valid once tau/tau_min >> 1 (here tau_min=600 s; already valid at 0.5 h = 1800 s).
    """
    return gamma_func(1.0 + beta) * TAU_MIN ** beta * tau ** (-beta)


def dacf_lognormal_theory(tau: np.ndarray, tau_R_mean: float, sigma: float,
                           n_mc: int = 50_000, seed: int = 999) -> np.ndarray:
    """E[exp(-tau/tau_R)] for log-normal tau_R by Monte Carlo (reference)."""
    rng = np.random.default_rng(seed)
    tau_samp = np.exp(rng.normal(np.log(tau_R_mean), sigma, n_mc))
    return np.exp(-np.outer(1.0 / tau_samp, tau)).mean(axis=0)


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_ensemble(D_R_vals: np.ndarray, seed_offset: int = 0) -> list:
    trajs = []
    for s, D_R_i in enumerate(D_R_vals):
        params = SimParams(D_T=D_T, D_R=D_R_i, v=V_BIO,
                           dt=DT_SIM, n_steps=N_STEPS, seed=seed_offset + s)
        trajs.append(Simulator(ActiveBrownianParticle(params), params).run())
    return trajs


def compute_msd(trajs: list) -> np.ndarray:
    return np.array([
        (tr[1:, 0] - tr[0, 0]) ** 2 + (tr[1:, 1] - tr[0, 1]) ** 2
        for tr in trajs
    ]).mean(axis=0)


def global_alpha(t: np.ndarray, msd: np.ndarray) -> float:
    return float(np.polyfit(np.log(t), np.log(msd), 1)[0])


def main():
    Pe = V_BIO ** 2 / (4.0 * D_T / TAU_R_MEAN)
    print("=== Heterogeneous ABP: Pareto tau_R distribution ===")
    print(f"V={V_BIO*3600:.0f} µm/h  D_T={D_T:.4f} µm^2/s  Pe(mean)={Pe:.2f}")
    print(f"tau_min={TAU_MIN:.0f} s = {TAU_MIN/60:.0f} min  dt={DT_SIM:.0f} s  "
          f"N_steps={N_STEPS} ({N_STEPS*DT_SIM/3600:.0f} h)  N_ens={N_ENSEMBLE}")
    print(f"Power law DACF ~ tau^(-beta) holds for tau >> tau_min = {TAU_MIN/3600:.2f} h")
    print()

    rng      = np.random.default_rng(seed=42)
    t_arr    = np.arange(1, N_STEPS + 1) * DT_SIM
    t_h      = t_arr / 3600.0
    max_lag  = N_STEPS // 2
    lag_s    = np.arange(max_lag) * DT_SIM
    lag_h    = lag_s / 3600.0

    results = {}
    for beta in BETA_VALUES:
        tau_R_vals = sample_pareto(beta, N_ENSEMBLE, rng)
        D_R_vals   = 1.0 / tau_R_vals

        print(f"beta={beta}  tau_R: median={np.median(tau_R_vals)/60:.0f} min  "
              f"90th pct={np.percentile(tau_R_vals, 90)/60:.0f} min  ...")

        trajs = run_ensemble(D_R_vals, seed_offset=int(beta * 1000))
        msd   = compute_msd(trajs)
        alpha = global_alpha(t_arr, msd)

        _, dacf_sim = orientation_acf(trajs, max_lag=max_lag)

        # Analytical DACF theory (power law)
        dacf_th = dacf_pareto_theory(lag_s[1:], beta)

        print(f"  global alpha = {alpha:.3f}")
        results[beta] = {
            "msd": msd, "dacf": dacf_sim, "dacf_th": dacf_th, "alpha": alpha,
        }

    # Log-normal reference (sigma=1.0, same mean as TAU_R_MEAN)
    print(f"Log-normal reference (sigma=1.0, mean=90 min)...")
    tau_R_ln   = np.exp(rng.normal(np.log(TAU_R_MEAN), 1.0, N_ENSEMBLE))
    D_R_ln     = 1.0 / tau_R_ln
    trajs_ln   = run_ensemble(D_R_ln, seed_offset=9999)
    msd_ln     = compute_msd(trajs_ln)
    alpha_ln   = global_alpha(t_arr, msd_ln)
    _, dacf_ln = orientation_acf(trajs_ln, max_lag=max_lag)
    dacf_ln_th = dacf_lognormal_theory(lag_s[1:], TAU_R_MEAN, 1.0)
    print(f"  global alpha = {alpha_ln:.3f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_dist, ax_msd, ax_dacf = axes

    # ── Panel 1: tau_R distributions ─────────────────────────────────────────
    tau_min_m = TAU_MIN / 60.0
    tau_max_m = 10000.0
    bins = np.logspace(np.log10(tau_min_m), np.log10(tau_max_m), 50)

    for beta in BETA_VALUES:
        tau_R_vals = sample_pareto(beta, N_ENSEMBLE, rng)
        ax_dist.hist(tau_R_vals / 60, bins=bins, density=True, alpha=0.35,
                     color=COLORS[beta], label=f"Pareto beta={beta}")

    # Overlay the power-law PDF slope guides
    t_pdf = np.logspace(np.log10(tau_min_m * 1.5), np.log10(tau_max_m * 0.5), 100)
    for beta in [0.83]:
        c = COLORS[beta]
        y_pdf = beta * (TAU_MIN / 60) ** beta * (t_pdf) ** (-(1 + beta))
        ax_dist.loglog(t_pdf, y_pdf, color=c, lw=2, ls="--",
                       label=f"Pareto PDF ~ tau^(-{1+beta:.2f})")

    # Log-normal reference
    tau_R_ln_plot = np.exp(rng.normal(np.log(TAU_R_MEAN), 1.0, N_ENSEMBLE))
    ax_dist.hist(tau_R_ln_plot / 60, bins=bins, density=True, alpha=0.35,
                 color="tab:blue", label="Log-normal sigma=1.0")

    ax_dist.set_xscale("log")
    ax_dist.set_yscale("log")
    ax_dist.axvline(TAU_MIN / 60, color="black", ls=":", lw=1.5,
                    label=f"tau_min = {TAU_MIN/60:.0f} min")
    ax_dist.axvline(TAU_R_MEAN / 60, color="gray", ls="--", lw=1.5,
                    label=f"tau_R_mean = {TAU_R_MEAN/60:.0f} min (log-normal)")
    ax_dist.set_xlabel("tau_R [min]")
    ax_dist.set_ylabel("probability density [min^-1]")
    ax_dist.set_title("tau_R distributions\nPareto (power-law tail) vs log-normal")
    ax_dist.legend(fontsize=7)
    ax_dist.grid(True, which="both", ls="--", alpha=0.3)

    # ── Panel 2: MSD ──────────────────────────────────────────────────────────
    t_ref = np.linspace(0.1, 10, 500)
    for beta in BETA_VALUES:
        r  = results[beta]
        lw = 2.5 if beta == 0.83 else 1.5
        ax_msd.loglog(t_h, r["msd"], color=COLORS[beta], lw=lw,
                      label=f"Pareto beta={beta}  alpha={r['alpha']:.2f}")

    ax_msd.loglog(t_h, msd_ln, color="tab:blue", lw=1.5, ls="--",
                  label=f"Log-normal sigma=1.0  alpha={alpha_ln:.2f}")
    ax_msd.loglog(t_ref, 193.92 * t_ref ** 1.40,
                  color="dimgray", lw=1.5, ls="--", alpha=0.8,
                  label="PDF 2D: 193.92*t^1.40")
    ax_msd.loglog(t_ref, 125.44 * t_ref ** 1.29,
                  color="gray", lw=1.2, ls="-.", alpha=0.7,
                  label="PDF 3D: 125.44*t^1.29")
    ax_msd.set_xlabel("time [h]")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("MSD — Pareto vs log-normal heterogeneous ABP")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 3: DACF log-log ─────────────────────────────────────────────────
    valid = lag_h[1:] > 0

    # Homogeneous ABP reference (exponential)
    dacf_homo = np.exp(-lag_s[1:] / TAU_R_MEAN)
    ax_dacf.loglog(lag_h[1:][valid], dacf_homo[valid],
                   color="black", lw=1.5, ls=":",
                   label=f"Homogeneous exp(-t/tau_R), tau_R=90 min")

    # Log-normal simulation + theory
    dacf_ln_cl = np.maximum(dacf_ln[1:], 1e-6)
    ax_dacf.loglog(lag_h[1:][valid], dacf_ln_cl[valid],
                   color="tab:blue", lw=1.5, ls="--",
                   label=f"Log-normal sigma=1.0 (sim) -- curved")
    ax_dacf.loglog(lag_h[1:][valid], np.maximum(dacf_ln_th[valid], 1e-6),
                   color="tab:blue", lw=1.0, ls=":", alpha=0.6,
                   label="Log-normal theory (MC integral)")

    # Pareto simulation + analytical theory
    for beta in BETA_VALUES:
        r  = results[beta]
        lw = 2.5 if beta == 0.83 else 1.5
        dacf_cl = np.maximum(r["dacf"][1:], 1e-6)
        ax_dacf.loglog(lag_h[1:][valid], dacf_cl[valid],
                       color=COLORS[beta], lw=lw,
                       label=f"Pareto beta={beta} (sim)")
        # Analytical power law (shown as dashed overlay)
        ax_dacf.loglog(lag_h[1:][valid],
                       np.maximum(r["dacf_th"][valid], 1e-6),
                       color=COLORS[beta], lw=lw * 0.5, ls=":",
                       alpha=0.7)

    # PDF measurement
    anchor_idx = np.argmin(np.abs(lag_h[1:] - 0.5))
    anchor_val = float(np.maximum(results[0.83]["dacf"][1:][anchor_idx], 1e-6))
    A_pl       = anchor_val / (0.5 ** (-0.83))
    t_pl       = lag_h[(lag_h > 0.2) & (lag_h < 5.5)]
    ax_dacf.loglog(t_pl, A_pl * t_pl ** (-0.83),
                   color="dimgray", lw=2.0, ls="--", alpha=0.9,
                   label="PDF measured ~ t^-0.83 (anchored)")

    ax_dacf.set_xlabel("lag time [h]")
    ax_dacf.set_ylabel("DACF  C_phi(tau)")
    ax_dacf.set_title("Orientation ACF (log-log)\n"
                      "Pareto -> straight line; log-normal -> curved")
    ax_dacf.legend(fontsize=7)
    ax_dacf.grid(True, which="both", ls="--", alpha=0.4)
    ax_dacf.set_ylim(1e-3, 1.2)

    fig.suptitle(
        f"Heterogeneous ABP: Pareto tau_R distribution\n"
        f"v={V_BIO*3600:.0f} µm/h, tau_min={TAU_MIN/60:.0f} min, "
        f"Pareto DACF ~ tau^(-beta) is exact power law (straight line on log-log)",
        fontsize=11,
    )
    fig.tight_layout()
    out = _HERE / "hetero_abp_pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    print("\n=== Summary ===")
    print(f"{'Distribution':<28} {'alpha(MSD)':<14} {'DACF shape'}")
    print("-" * 60)
    for beta in BETA_VALUES:
        print(f"Pareto beta={beta:<18.2f} {results[beta]['alpha']:<14.3f} power law ~ t^-{beta}")
    print(f"Log-normal sigma=1.0         {alpha_ln:<14.3f} stretched exponential (not power law)")
    print()
    print(f"Target: alpha = 1.40,  DACF ~ t^-0.83")
    print(f"Pareto beta=0.83 gives exact power-law DACF with slope -0.83")


if __name__ == "__main__":
    main()
