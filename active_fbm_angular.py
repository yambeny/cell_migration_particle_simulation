"""Active FBM with correlated angular noise (fGn on both translation AND rotation).

Extends active_fbm.py by replacing the white angular noise with fractional
Gaussian noise parameterised by H_phi:

    x(t+dt)   = x(t)   + v*cos(phi)*dt  +  xi_H_x(t)         [fGn, H_trans]
    y(t+dt)   = y(t)   + v*sin(phi)*dt  +  xi_H_y(t)          [fGn, H_trans]
    phi(t+dt) = phi(t) + sqrt(2*D_R*dt) * zeta_H_phi(t)       [fGn, H_phi]

where xi_H and zeta are independent fGn sequences, both normalised to the
same per-step variance as the standard white-noise versions.

Physical motivation for correlated angular noise
-------------------------------------------------
H_phi < 0.5  (ANTIPERSISTENT):  if the cell was turning clockwise, the next
    increment is more likely counter-clockwise.  Net rotation accumulates as
    a sub-diffusive fBM: Var(dphi(tau)) ~ tau^{2*H_phi}, with 2*H_phi < 1.
    DACF = exp(-D_R * tau^{2*H_phi} * dt^{1-2*H_phi})
    -> stretched exponential that decays SLOWER than the standard exponential.

H_phi = 0.5  (standard white noise, baseline):
    DACF = exp(-D_R * tau)  ->  exponential with tau_R = 90 min.

H_phi > 0.5  (persistent): cell keeps turning in the same direction -> phi
    diffuses FASTER -> DACF decays faster than exponential.  Wrong direction.

Target: DACF ~ tau^{-0.83} (measured 2D, Sde Boker PDF Jan 2026).
    The stretched exponential approximates this power law in a log-log window.
    Numerical calibration (see below): H_phi ≈ 0.46 gives average log-log
    slope ≈ -0.83 over the 0.5–5 h observation window.

Normalisation (same per-step variance for all H)
-------------------------------------------------
Translational noise scale:  sqrt(2 * D_T  * dt)   -- all H_trans
Angular noise scale:        sqrt(2 * D_R  * dt)   -- all H_phi
Correlations are encoded entirely in the Cholesky factor of the Toeplitz
covariance matrix; the per-step amplitude is unchanged.  This isolates the
effect of memory from the effect of amplitude.

Reference: docs/endoderm_migration_models.md
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import cholesky, toeplitz

from simulation.params import SimParams
from simulation.analysis import orientation_acf

_HERE = Path(__file__).parent

# ── Physical parameters (same as active_fbm.py / hetero_abp.py) ──────────────
RADIUS_UM  = 6.5
T_K        = 310.0
ETA_PA_S   = 1e-3
V_BIO      = 0.0083    # µm/s = 30 µm/h
TAU_R_MEAN = 5400.0    # s  = 90 min

_phys   = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
D_T     = _phys.D_T
D_R_BIO = 1.0 / TAU_R_MEAN

# ── Simulation parameters ──────────────────────────────────────────────────────
DT_SIM     = 60.0   # s
N_STEPS    = 600    # 10 h
N_ENSEMBLE = 300    # more particles for smoother DACF

# ── fGn generation ─────────────────────────────────────────────────────────────

def _fgn_cov(n, H):
    k   = np.arange(n, dtype=float)
    rho = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                 + np.maximum(k - 1, 0) ** (2 * H))
    rho[0] = 1.0
    return rho


def _cholesky_factor(n, H):
    return cholesky(toeplitz(_fgn_cov(n, H)), lower=True)


def sample_fgn(L, size, rng):
    """Draw `size` independent unit-variance fGn sequences of length n."""
    n = L.shape[0]
    Z = rng.standard_normal((n, size))
    return (L @ Z).T   # (size, n)


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(H_trans, H_phi, rng, n_ensemble=N_ENSEMBLE):
    """Run n_ensemble trajectories with fGn translation (H_trans) and rotation (H_phi).

    Normalisation: per-step variance is the same for all H (same as white noise).
        Translational noise per step: sqrt(2 * D_T * dt)
        Angular noise per step:       sqrt(2 * D_R * dt)
    Correlations are captured by the Cholesky factor alone.

    Returns:
        pos  : (N_STEPS+1, n_ensemble, 2) unwrapped positions
        phis : (N_STEPS+1, n_ensemble) orientations
    """
    dt = DT_SIM
    n  = N_STEPS

    # Pre-compute Cholesky factors
    L_trans = _cholesky_factor(n, H_trans)
    L_phi   = _cholesky_factor(n, H_phi)

    # Noise scales: same per-step variance as white noise for all H
    scale_trans = np.sqrt(2.0 * D_T    * dt)   # µm
    scale_phi   = np.sqrt(2.0 * D_R_BIO * dt)  # rad

    xi_x = sample_fgn(L_trans, n_ensemble, rng) * scale_trans  # (n_ens, n)
    xi_y = sample_fgn(L_trans, n_ensemble, rng) * scale_trans
    zeta = sample_fgn(L_phi,   n_ensemble, rng) * scale_phi    # (n_ens, n)

    # Initial conditions
    x   = np.zeros(n_ensemble)
    y   = np.zeros(n_ensemble)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_ensemble)

    pos  = np.empty((n + 1, n_ensemble, 2))
    phis = np.empty((n + 1, n_ensemble))
    pos[0, :, 0] = x
    pos[0, :, 1] = y
    phis[0]      = phi

    for t in range(n):
        x   += V_BIO * np.cos(phi) * dt + xi_x[:, t]
        y   += V_BIO * np.sin(phi) * dt + xi_y[:, t]
        phi += zeta[:, t]
        pos[t + 1, :, 0] = x
        pos[t + 1, :, 1] = y
        phis[t + 1]       = phi

    return pos, phis


# ── Analysis helpers ───────────────────────────────────────────────────────────

def compute_msd(pos):
    dx = pos[1:, :, 0] - pos[0:1, :, 0]
    dy = pos[1:, :, 1] - pos[0:1, :, 1]
    return (dx ** 2 + dy ** 2).mean(axis=1)


def compute_dacf(phis, max_lag):
    trajs = [
        np.column_stack([np.zeros((phis.shape[0], 2)), phis[:, i]])
        for i in range(phis.shape[1])
    ]
    _, acf = orientation_acf(trajs, max_lag=max_lag)
    return acf


def global_alpha(t, msd):
    return float(np.polyfit(np.log(t), np.log(msd), 1)[0])


def dacf_slope(lag, dacf, t_min_h=0.5, t_max_h=5.0):
    """Fit power-law slope to DACF in the log-log window [t_min_h, t_max_h]."""
    mask = (lag > t_min_h) & (lag < t_max_h) & (dacf > 1e-4)
    if mask.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log(lag[mask]), np.log(dacf[mask]), 1)[0])


# ── Cases ──────────────────────────────────────────────────────────────────────
# H_phi < 0.5: antipersistent rotation -> slower DACF decay (correct direction)
# H_phi = 0.5: standard white noise (reference)
# H_phi ~ 0.46: calibrated to give average log-log DACF slope ~ -0.83
CASES = [
    (0.5, 0.50, "ABP standard (H_t=0.5, H_phi=0.50)",       "tab:blue"),
    (0.7, 0.50, "Trans. fBM only (H_t=0.7, H_phi=0.50)",    "tab:cyan"),
    (0.5, 0.46, "Ang. fBM only (H_t=0.5, H_phi=0.46)",      "tab:orange"),
    (0.7, 0.46, "Full model (H_t=0.7, H_phi=0.46)",          "tab:red"),
    (0.7, 0.48, "Full model (H_t=0.7, H_phi=0.48)",          "tab:purple"),
]


def main():
    Pe = V_BIO ** 2 / (4.0 * D_T * D_R_BIO)
    print("=== Active FBM: correlated angular noise ===")
    print(f"V={V_BIO*3600:.0f} µm/h  tau_R={TAU_R_MEAN/60:.0f} min  "
          f"D_T={D_T:.4f} µm^2/s  Pe={Pe:.2f}")
    print(f"dt={DT_SIM:.0f} s  N_steps={N_STEPS} ({N_STEPS*DT_SIM/3600:.0f} h)  "
          f"N_ens={N_ENSEMBLE}")
    print(f"H_phi < 0.5 = antipersistent rotation (slower DACF decay, correct direction)")
    print(f"H_phi ~ 0.46 targets DACF slope ~ -0.83 over 0.5-5 h window")
    print()

    rng     = np.random.default_rng(seed=42)
    t_arr   = np.arange(1, N_STEPS + 1) * DT_SIM
    t_h     = t_arr / 3600.0
    max_lag = N_STEPS // 2
    lag_h   = np.arange(max_lag) * DT_SIM / 3600.0

    results = {}
    for (H_t, H_p, label, col) in CASES:
        key = (H_t, H_p)
        print(f"Running {label} ...")
        pos, phis = run_simulation(H_t, H_p, rng)
        msd  = compute_msd(pos)
        dacf = compute_dacf(phis, max_lag)
        alph = global_alpha(t_arr, msd)
        sl   = dacf_slope(lag_h[1:], dacf[1:])
        print(f"  alpha(MSD) = {alph:.3f}   DACF slope = {sl:.3f}")
        results[key] = {
            "msd": msd, "dacf": dacf, "alpha": alph,
            "dacf_slope": sl, "label": label, "color": col,
        }

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_msd, ax_dacf_lin, ax_dacf = axes

    t_ref  = np.linspace(0.1, 10, 500)
    dacf_th_exp = np.exp(-D_R_BIO * np.arange(max_lag) * DT_SIM)

    # ── Panel 1: MSD ─────────────────────────────────────────────────────────
    for (H_t, H_p, label, col) in CASES:
        r  = results[(H_t, H_p)]
        lw = 2.5 if H_p == 0.5 else 1.8
        ax_msd.loglog(t_h, r["msd"], color=col, lw=lw,
                      label=f"{label}  a={r['alpha']:.2f}")
    ax_msd.loglog(t_ref, 193.92 * t_ref ** 1.40,
                  color="dimgray", lw=1.5, ls="--", alpha=0.8,
                  label="PDF 2D: 193.92*t^1.40")
    ax_msd.loglog(t_ref, 125.44 * t_ref ** 1.29,
                  color="gray", lw=1.2, ls="-.", alpha=0.7,
                  label="PDF 3D: 125.44*t^1.29")
    ax_msd.set_xlabel("time [h]")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("MSD: correlated angular noise\n"
                     "H_phi < 0.5 increases effective persistence -> higher alpha")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 2: DACF linear scale ───────────────────────────────────────────
    plot_lag = lag_h <= 8.0
    ax_dacf_lin.plot(lag_h[plot_lag], dacf_th_exp[plot_lag],
                     color="black", lw=1.5, ls=":", label="Exp. theory (H_phi=0.5)")
    for (H_t, H_p, label, col) in CASES:
        r = results[(H_t, H_p)]
        ax_dacf_lin.plot(lag_h[plot_lag], r["dacf"][plot_lag],
                         color=col, lw=1.8, label=label)
    ax_dacf_lin.set_xlabel("lag time [h]")
    ax_dacf_lin.set_ylabel("DACF")
    ax_dacf_lin.set_title("Orientation ACF (linear scale)")
    ax_dacf_lin.legend(fontsize=7)
    ax_dacf_lin.grid(True, ls="--", alpha=0.4)
    ax_dacf_lin.set_ylim(-0.05, 1.05)

    # ── Panel 3: DACF log-log ────────────────────────────────────────────────
    valid = lag_h[1:] > 0
    ax_dacf.loglog(lag_h[1:][valid], np.maximum(dacf_th_exp[1:][valid], 1e-5),
                   color="black", lw=1.5, ls=":", label="Exp. theory (H_phi=0.5)")
    for (H_t, H_p, label, col) in CASES:
        r    = results[(H_t, H_p)]
        dacf = np.maximum(r["dacf"][1:], 1e-5)
        lw   = 2.5 if H_p == 0.5 else 1.8
        sl   = r["dacf_slope"]
        ax_dacf.loglog(lag_h[1:][valid], dacf[valid],
                       color=col, lw=lw,
                       label=f"{label}  slope={sl:.2f}")

    # PDF power-law reference anchored at t=0.5h using the standard ABP DACF
    ref_dacf = np.maximum(results[(0.5, 0.5)]["dacf"][1:], 1e-5)
    anchor   = np.argmin(np.abs(lag_h[1:] - 0.5))
    A_pl     = ref_dacf[anchor] / (0.5 ** (-0.83))
    t_pl     = lag_h[(lag_h > 0.2) & (lag_h < 5.5)]
    ax_dacf.loglog(t_pl, A_pl * t_pl ** (-0.83),
                   color="dimgray", lw=2.0, ls="--", alpha=0.9,
                   label="PDF measured: ~ t^-0.83")

    ax_dacf.set_xlabel("lag time [h]")
    ax_dacf.set_ylabel("DACF  C_phi(tau)")
    ax_dacf.set_title("Orientation ACF (log-log)\n"
                      "H_phi=0.46 targets slope = -0.83")
    ax_dacf.legend(fontsize=7)
    ax_dacf.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle(
        f"Active FBM + antipersistent angular noise (H_phi < 0.5)\n"
        f"v={V_BIO*3600:.0f} µm/h, tau_R={TAU_R_MEAN/60:.0f} min, "
        f"all H use same per-step noise variance as white noise",
        fontsize=11,
    )
    fig.tight_layout()
    out = _HERE / "active_fbm_angular.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    print("\n=== Summary ===")
    print(f"{'H_trans':<10} {'H_phi':<10} {'alpha(MSD)':<14} {'DACF slope':<14}  label")
    print("-" * 80)
    for (H_t, H_p, label, _) in CASES:
        r = results[(H_t, H_p)]
        print(f"{H_t:<10.2f} {H_p:<10.2f} {r['alpha']:<14.3f} {r['dacf_slope']:<14.3f}  {label}")
    print()
    print("Target: alpha(MSD) ~ 1.40,  DACF slope ~ -0.83")


if __name__ == "__main__":
    main()
