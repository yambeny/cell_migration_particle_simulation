"""Heterogeneous ABP simulation: endoderm migration model.

Compares homogeneous vs heterogeneous ABP populations using biological parameters
extracted from the Sde Boker PDF (Jan 2026). Each cell is a standard ABP, but
tau_R is drawn from a log-normal distribution reflecting cell-to-cell variability
in cytoskeletal persistence.

Key numbers (from PDF):
  v      = 30 µm/h = 0.0083 µm/s  (velocity histogram)
  tau_R  = 90 min  (cytoskeletal memory window; NOT Stokes-Einstein)
  D_T    = Stokes-Einstein (thermal; unchanged)
  Pe     = v^2 / (4 D_T D_R) ~ 2.68  (> 1, so ballistic regime exists)

Expected outcome:
  Single ABP, biological params:     global alpha ~ 1.39
  Heterogeneous ABP (sigma_log=1.0): global alpha ~ 1.40  (matches 2D measurement)

Reference: docs/endoderm_migration_models.md
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulation.params import SimParams
from simulation.particle import ActiveBrownianParticle
from simulation.simulator import Simulator
from simulation.analysis import orientation_acf
from simulation.theory import (
    active_msd, active_msd_exponent,
    effective_diffusion,
)

_HERE = Path(__file__).parent

# ── Biological parameters (Sde Boker PDF Jan 2026) ────────────────────────────
RADIUS_UM      = 6.5       # µm  (nucleus area 73.3 µm^2 → R_nuc≈4.8 µm; cell R≈6-7 µm)
T_K            = 310.0     # K  (37°C physiological)
ETA_PA_S       = 1e-3      # Pa·s  (water-like culture medium)
V_BIO          = 0.0083    # µm/s  (= 30 µm/h; from PDF velocity histogram)
TAU_R_MEAN     = 5400.0    # s  (= 90 min; biological persistence from PDF memory window)
# ── Population heterogeneity ───────────────────────────────────────────────────
# sigma_log = spread of ln(tau_R).  sigma=1.0 → 5th-95th pct ≈ 19-519 min
SIGMA_VALUES   = [0.0, 0.5, 1.0, 1.5]

# ── Simulation timing ──────────────────────────────────────────────────────────
# Window: 10 h (matches PDF observation window).
# dt=60 s → D_R_mean * dt = 1/5400 * 60 ≈ 0.011 << 1  (time step OK)
DT_SIM         = 60.0      # s  (1 min steps)
N_STEPS_MSD    = 600       # 600 * 60 s = 36000 s = 10 h
N_ENSEMBLE     = 200       # particles per sigma value

# ── Derived D_T (Stokes-Einstein) ─────────────────────────────────────────────
_phys  = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
D_T    = _phys.D_T
D_R_SE = _phys.D_R          # Stokes-Einstein D_R (for comparison)
D_R_BIO = 1.0 / TAU_R_MEAN  # biological D_R (set by cytoskeletal persistence)

# ── Plot colours (consistent across figures) ──────────────────────────────────
_COLORS = {0.0: "tab:blue", 0.5: "tab:orange", 1.0: "tab:red", 1.5: "tab:purple"}
_LABELS = {
    0.0: "Homogeneous  (sigma=0)",
    0.5: "Heterogeneous  sigma=0.5",
    1.0: "Heterogeneous  sigma=1.0",
    1.5: "Heterogeneous  sigma=1.5",
}


# ── Simulation helpers ─────────────────────────────────────────────────────────

def _run_particle(D_R_i: float, seed: int) -> np.ndarray:
    params = SimParams(
        D_T=D_T, D_R=D_R_i, v=V_BIO,
        dt=DT_SIM, n_steps=N_STEPS_MSD, seed=seed,
    )
    return Simulator(ActiveBrownianParticle(params), params).run()


def run_ensemble(D_R_values: np.ndarray, seed_offset: int = 0) -> list[np.ndarray]:
    return [
        _run_particle(D_R_i, seed_offset + s)
        for s, D_R_i in enumerate(D_R_values)
    ]


def compute_msd(trajs: list[np.ndarray]) -> np.ndarray:
    return np.array([
        (tr[1:, 0] - tr[0, 0]) ** 2 + (tr[1:, 1] - tr[0, 1]) ** 2
        for tr in trajs
    ]).mean(axis=0)


def global_alpha_fit(t: np.ndarray, msd: np.ndarray) -> float:
    """Global power-law exponent: MSD ~ t^alpha over the full window."""
    return float(np.polyfit(np.log(t), np.log(msd), 1)[0])


def theoretical_dacf_hetero(t: np.ndarray, tau_R_mean: float, sigma_log: float,
                             n_mc: int = 20_000) -> np.ndarray:
    """E[exp(-t/tau_R)] for log-normal tau_R, computed by Monte Carlo.

    This is the expected ensemble-average DACF when each cell has its own tau_R.
    Result is a stretched-exponential / power-law-like decay — slower than any
    individual cell's exponential.
    """
    rng = np.random.default_rng(seed=999)
    tau_samp = np.exp(rng.normal(np.log(tau_R_mean), sigma_log, n_mc))
    # outer product: (n_mc, len(t)) → each row is exp(-t/tau_R_i)
    return np.exp(-np.outer(1.0 / tau_samp, t)).mean(axis=0)


def main() -> None:
    # ── Print key dimensionless numbers ───────────────────────────────────────
    Pe_bio   = V_BIO ** 2 / (4.0 * D_T * D_R_BIO)
    tau_T    = 4.0 * D_T / V_BIO ** 2   # diffusive→ballistic crossover
    Pe_SE    = V_BIO ** 2 / (4.0 * D_T * D_R_SE)
    print(f"D_T = {D_T:.4f} µm^2/s   (Stokes-Einstein)")
    print(f"D_R_SE  = {D_R_SE:.2e} rad^2/s   tau_R_SE  = {1/D_R_SE/60:.0f} min   Pe_SE  = {Pe_SE:.2f}")
    print(f"D_R_BIO = {D_R_BIO:.2e} rad^2/s   tau_R_BIO = {TAU_R_MEAN/60:.0f} min   Pe_BIO = {Pe_bio:.2f}")
    print(f"V_BIO = {V_BIO:.4f} µm/s = {V_BIO*3600:.1f} µm/h")
    print(f"tau_T = {tau_T:.0f} s = {tau_T/60:.0f} min")
    print(f"Ballistic regime (tau_T < tau_R_BIO): {'YES' if tau_T < TAU_R_MEAN else 'NO'}")
    print()

    # ── Draw tau_R distributions and run ensembles ────────────────────────────
    rng = np.random.default_rng(seed=42)
    t_arr = np.arange(1, N_STEPS_MSD + 1) * DT_SIM   # [s]
    t_h   = t_arr / 3600.0                             # [h]

    ensembles: dict[float, dict] = {}
    for sigma in SIGMA_VALUES:
        if sigma == 0.0:
            D_R_vals = np.full(N_ENSEMBLE, D_R_BIO)
        else:
            tau_samp = np.exp(rng.normal(np.log(TAU_R_MEAN), sigma, N_ENSEMBLE))
            D_R_vals = 1.0 / tau_samp

        print(f"Running sigma={sigma}  ({N_ENSEMBLE} particles × {N_STEPS_MSD} steps)...")
        seed_off = int(sigma * 1000)
        trajs = run_ensemble(D_R_vals, seed_offset=seed_off)
        msd   = compute_msd(trajs)
        alpha = global_alpha_fit(t_arr, msd)
        print(f"  global alpha = {alpha:.4f}")

        ensembles[sigma] = {
            "trajs":    trajs,
            "msd":      msd,
            "alpha":    alpha,
            "D_R_vals": D_R_vals,
        }
    print()

    # ── DACF computation (lag up to half trajectory length for clean statistics)
    max_dacf_lag = N_STEPS_MSD // 2   # 300 steps = 5 h
    lag_steps    = np.arange(max_dacf_lag)
    lag_s        = lag_steps * DT_SIM  # [s]
    lag_h        = lag_s / 3600.0      # [h]

    dacf_computed: dict[float, np.ndarray] = {}
    for sigma in [0.0, 1.0, 1.5]:
        print(f"Computing DACF for sigma={sigma}...")
        _, acf = orientation_acf(ensembles[sigma]["trajs"], max_lag=max_dacf_lag)
        dacf_computed[sigma] = acf

    # ── Figures ───────────────────────────────────────────────────────────────
    # Figure 1: 2×2 summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_msd, ax_alpha = axes[0]
    ax_dacf, ax_dist = axes[1]

    # ── Panel 1: MSD log-log ──────────────────────────────────────────────────
    for sigma in SIGMA_VALUES:
        e  = ensembles[sigma]
        lw = 2.5 if sigma in (0.0, 1.0) else 1.5
        ls = "-" if sigma in (0.0, 1.0) else "--"
        ax_msd.loglog(
            t_h, e["msd"], color=_COLORS[sigma], lw=lw, ls=ls,
            label=f"{_LABELS[sigma]}  (alpha={e['alpha']:.3f})",
        )

    # Homogeneous theory (exact ABP formula)
    msd_th_bio = active_msd(t_arr, D_T, D_R_BIO, V_BIO)
    ax_msd.loglog(t_h, msd_th_bio, color="tab:blue", lw=1.2, ls=":", alpha=0.6,
                  label="Homogeneous exact theory")

    # Measured power laws from PDF (fitted to actual data in the paper)
    t_ref = np.linspace(0.1, 10.0, 500)
    ax_msd.loglog(t_ref, 125.44 * t_ref ** 1.29,
                  color="black", lw=1.5, ls="-.", alpha=0.8,
                  label="Measured: 125.44 * t^1.29  (3D micropattern)")
    ax_msd.loglog(t_ref, 193.92 * t_ref ** 1.40,
                  color="dimgray", lw=1.5, ls="--", alpha=0.8,
                  label="Measured: 193.92 * t^1.40  (2D)")

    ax_msd.set_xlabel("time [h]")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("MSD — homogeneous vs heterogeneous ABP\nvs measured endoderm migration (Sde Boker PDF)")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 2: global alpha vs sigma ────────────────────────────────────────
    sigma_arr = np.array(SIGMA_VALUES)
    alpha_arr = np.array([ensembles[s]["alpha"] for s in SIGMA_VALUES])
    ax_alpha.plot(sigma_arr, alpha_arr, "o-", color="tab:blue", lw=2, ms=9,
                  label="Simulation global alpha  (MSD ~ t^alpha over 0-10 h)")
    for s, a in zip(SIGMA_VALUES, alpha_arr):
        ax_alpha.annotate(f"{a:.3f}", (s, a), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=9)
    ax_alpha.axhline(1.40, color="dimgray", ls="--", lw=1.5,
                     label="Measured alpha = 1.40  (2D)")
    ax_alpha.axhline(1.29, color="black", ls="-.", lw=1.5,
                     label="Measured alpha = 1.29  (3D micropattern)")
    ax_alpha.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5,
                     label="alpha = 1  (normal diffusion)")
    ax_alpha.set_xlabel("sigma_log  (spread of ln tau_R)")
    ax_alpha.set_ylabel("global alpha  (MSD ~ t^alpha)")
    ax_alpha.set_title("Global MSD exponent vs population heterogeneity")
    ax_alpha.set_xticks(SIGMA_VALUES)
    ax_alpha.set_ylim(1.0, 1.65)
    ax_alpha.legend(fontsize=9)
    ax_alpha.grid(True, ls="--", alpha=0.4)

    # ── Panel 3: ensemble DACF (log-log) ─────────────────────────────────────
    # Skip lag=0 (t=0, undefined on log axis).  Show sigma=0, 1.0, 1.5.
    t_plot = lag_h[1:]   # start from lag=1

    for sigma in [0.0, 1.0, 1.5]:
        acf = np.maximum(dacf_computed[sigma][1:], 1e-6)
        ax_dacf.loglog(t_plot, acf, color=_COLORS[sigma], lw=1.8,
                       label=f"sim sigma={sigma}")

    # Theoretical homogeneous: straight line on log-log (exponential)
    acf_th_homo = np.exp(-D_R_BIO * lag_s[1:])
    ax_dacf.loglog(t_plot, acf_th_homo, color="tab:blue", lw=1.2, ls=":", alpha=0.7,
                   label=f"Theory sigma=0: exp(-t/tau_R),  tau_R={TAU_R_MEAN/60:.0f} min")

    # Theoretical heterogeneous sigma=1.0: E[exp(-t/tau_R)]
    acf_th_hetero = np.maximum(
        theoretical_dacf_hetero(lag_s[1:], TAU_R_MEAN, 1.0), 1e-6
    )
    ax_dacf.loglog(t_plot, acf_th_hetero, color="tab:red", lw=1.2, ls=":", alpha=0.7,
                   label="Theory sigma=1.0: E[exp(-t/tau_R)]  (MC integral)")

    # PDF measured power-law reference: DACF ~ t^{-0.83} for 2D
    # Anchored to match DACF at t=0.5 h for visual alignment
    t_ref_dacf = t_plot[(t_plot > 0.3) & (t_plot < 5.0)]
    anchor_idx = np.argmin(np.abs(t_plot - 0.5))
    anchor_val = float(np.maximum(dacf_computed[1.0][1:][anchor_idx], 1e-6))
    A_pl = anchor_val / (0.5 ** (-0.83))
    ax_dacf.loglog(t_ref_dacf, A_pl * t_ref_dacf ** (-0.83),
                   color="black", lw=1.5, ls="--", alpha=0.7,
                   label="PDF measured: ~ t^-0.83  (2D, anchored at t=0.5 h)")

    ax_dacf.set_xlabel("lag time [h]")
    ax_dacf.set_ylabel("DACF  C_phi(tau)")
    ax_dacf.set_title("Ensemble orientation ACF — exponential vs non-exponential decay\n"
                      "Heterogeneous tau_R produces apparent power-law DACF")
    ax_dacf.legend(fontsize=7.5)
    ax_dacf.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 4: tau_R distribution for sigma=1.0 ────────────────────────────
    tau_R_min_m = TAU_R_MEAN * np.exp(-3.5 * 1.0)
    tau_R_max_m = TAU_R_MEAN * np.exp(3.5 * 1.0)
    tau_R_samp_min = 5.0      # 5 min floor
    tau_R_samp_max = 2000.0   # 2000 min ceiling for visibility

    tau_R_min = max(tau_R_min_m / 60.0, tau_R_samp_min)
    tau_R_max = min(tau_R_max_m / 60.0, tau_R_samp_max)
    bins_log = np.logspace(np.log10(tau_R_min), np.log10(tau_R_max), 30)

    tau_R_vals_min = 1.0 / ensembles[1.0]["D_R_vals"] / 60.0  # minutes

    ax_dist.hist(tau_R_vals_min, bins=bins_log, density=True, alpha=0.7,
                 color="tab:red", label=f"Drawn tau_R  (N={N_ENSEMBLE}, sigma=1.0)")
    ax_dist.axvline(TAU_R_MEAN / 60.0, color="black", ls="--", lw=1.8,
                    label=f"Mean = {TAU_R_MEAN/60:.0f} min")
    pct5  = float(np.percentile(tau_R_vals_min, 5))
    pct95 = float(np.percentile(tau_R_vals_min, 95))
    ax_dist.axvline(pct5,  color="gray", ls=":", lw=1.5,
                    label=f"5th pct = {pct5:.0f} min")
    ax_dist.axvline(pct95, color="gray", ls="-.", lw=1.5,
                    label=f"95th pct = {pct95:.0f} min")
    ax_dist.set_xscale("log")
    ax_dist.set_xlabel("tau_R [min]")
    ax_dist.set_ylabel("probability density [min^-1]")
    ax_dist.set_title(f"tau_R distribution  (sigma_log = 1.0)\n"
                      f"5th-95th percentile: {pct5:.0f} - {pct95:.0f} min")
    ax_dist.legend(fontsize=9)
    ax_dist.grid(True, ls="--", alpha=0.4)

    Pe_bio_val = V_BIO ** 2 / (4.0 * D_T * D_R_BIO)
    fig.suptitle(
        f"Heterogeneous ABP: endoderm migration model\n"
        f"v = {V_BIO*3600:.0f} µm/h,  tau_R_mean = {TAU_R_MEAN/60:.0f} min,  "
        f"D_T = {D_T:.4f} µm^2/s,  Pe = {Pe_bio_val:.2f}  "
        f"(biological params from Sde Boker PDF Jan 2026)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(_HERE / "hetero_abp.png", dpi=150)
    plt.close(fig)
    print("Saved hetero_abp.png")

    # ── Figure 2: local exponent alpha(t) — Stokes-Einstein vs biological D_R ─
    # Shows why the biological D_R (3.3x larger tau_R) is critical for getting
    # alpha > 1.  Same v, same D_T; only D_R changes.
    fig2, ax_exp = plt.subplots(figsize=(10, 5))

    tau_R_SE_s = 1.0 / D_R_SE
    Pe_SE_val  = V_BIO ** 2 / (4.0 * D_T * D_R_SE)

    alpha_bio = active_msd_exponent(t_arr, D_T, D_R_BIO, V_BIO)
    alpha_SE  = active_msd_exponent(t_arr, D_T, D_R_SE,  V_BIO)

    peak_bio  = float(alpha_bio.max())
    peak_SE   = float(alpha_SE.max())
    t_peak_bio = float(t_arr[np.argmax(alpha_bio)])
    t_peak_SE  = float(t_arr[np.argmax(alpha_SE)])

    glob_bio = global_alpha_fit(t_arr, active_msd(t_arr, D_T, D_R_BIO, V_BIO))
    glob_SE  = global_alpha_fit(t_arr, active_msd(t_arr, D_T, D_R_SE,  V_BIO))

    ax_exp.semilogx(
        t_h, alpha_bio, color="tab:red", lw=2.5,
        label=(
            f"Biological D_R  (tau_R={TAU_R_MEAN/60:.0f} min, Pe={Pe_bio_val:.2f})\n"
            f"  peak alpha={peak_bio:.3f} at t={t_peak_bio/60:.0f} min, "
            f"global alpha={glob_bio:.3f}"
        ),
    )
    ax_exp.semilogx(
        t_h, alpha_SE, color="tab:blue", lw=2.5, ls="--",
        label=(
            f"Stokes-Einstein D_R  (tau_R={tau_R_SE_s/60:.0f} min, Pe={Pe_SE_val:.2f})\n"
            f"  peak alpha={peak_SE:.3f} at t={t_peak_SE/60:.0f} min, "
            f"global alpha={glob_SE:.3f}"
        ),
    )
    ax_exp.axhline(1.40, color="dimgray", ls="--", lw=1.5,
                   label="Measured alpha = 1.40  (2D)")
    ax_exp.axhline(1.29, color="black",   ls="-.", lw=1.5,
                   label="Measured alpha = 1.29  (3D micropattern)")
    ax_exp.axhline(1.0,  color="gray",    ls=":",  lw=1, alpha=0.5)
    ax_exp.axhline(2.0,  color="gray",    ls="--", lw=1, alpha=0.5,
                   label="alpha = 2  (ballistic)")
    ax_exp.axvline(TAU_R_MEAN  / 3600, color="tab:red",  ls=":", lw=1, alpha=0.5,
                   label=f"tau_R bio = {TAU_R_MEAN/60:.0f} min")
    ax_exp.axvline(tau_R_SE_s  / 3600, color="tab:blue", ls=":", lw=1, alpha=0.5,
                   label=f"tau_R SE  = {tau_R_SE_s/60:.0f} min")

    ax_exp.set_xlabel("time [h]")
    ax_exp.set_ylabel("local exponent  alpha(t) = d(log MSD) / d(log t)")
    ax_exp.set_title(
        "MSD local exponent: Stokes-Einstein D_R vs biological D_R\n"
        f"Same v={V_BIO*3600:.0f} µm/h, same D_T.  "
        f"Biological tau_R is {TAU_R_MEAN/tau_R_SE_s:.1f}x longer → Pe flips from <1 to >1."
    )
    ax_exp.set_ylim(0.8, 2.3)
    ax_exp.legend(fontsize=9)
    ax_exp.grid(True, which="both", ls="--", alpha=0.4)

    fig2.tight_layout()
    fig2.savefig(_HERE / "hetero_abp_exponent.png", dpi=150)
    plt.close(fig2)
    print("Saved hetero_abp_exponent.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'sigma_log':<12} {'global alpha':<14} {'vs 1.40 (2D)':<16} {'vs 1.29 (3D)'}")
    print("-" * 56)
    for sigma in SIGMA_VALUES:
        a = ensembles[sigma]["alpha"]
        print(f"{sigma:<12.1f} {a:<14.4f} {a - 1.40:+.4f}          {a - 1.29:+.4f}")
    print()
    print(f"sigma_log=1.0 matches 2D measured alpha=1.40: "
          f"{'YES' if abs(ensembles[1.0]['alpha'] - 1.40) < 0.02 else 'NO'}")


if __name__ == "__main__":
    main()
