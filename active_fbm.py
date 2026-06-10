"""Active Fractional Brownian Motion (Active FBM) simulation.

Standard ABP translational noise (white) is replaced with fractional Gaussian
noise (fGn) whose increments have long-range temporal correlations parameterised
by the Hurst exponent H:

    x(t+dt) = x(t) + v*cos(phi)*dt  +  xi_H_x(t)
    y(t+dt) = y(t) + v*sin(phi)*dt  +  xi_H_y(t)
    phi(t+dt) = phi(t) + sqrt(2*D_R*dt)*eta          [standard, unchanged]

where xi_H has covariance <xi_H(t) xi_H(s)> ~ D_H*|t-s|^{2H-2} (power-law memory).

H=0.5 : white noise  ->  standard ABP (normal diffusion at long times)
H>0.5 : persistent   ->  superdiffusion, MSD ~ t^{2H}
H=0.7 : MSD ~ t^1.4  ->  matches measured endoderm 2D exponent

Physical motivation: GLE (Generalised Langevin Equation) with power-law memory
kernel gamma(t) ~ t^{-(1-beta)}, beta = 2H-1. Arises when the cell crawls on a
viscoelastic ECM whose stress-relaxation is a power law (measured in collagen/
fibronectin gels).

What Active FBM DOES reproduce:
    MSD ~ t^{2H}  (correct exponent for H=0.7)
What it does NOT reproduce:
    DACF is still exponential (angular process is unchanged).
    Power-law DACF requires correlated angular noise -- a different model.

Analytical MSD (exact, both contributions independent):
    MSD(t) = 2*D_H*t^{2H}  +  (v^2/D_R)*(2t - 2*tau_R*(1-exp(-t/tau_R)))
                fBM term              active swim term

fGn generation: Cholesky decomposition of the Toeplitz covariance matrix.
O(N^3) factorisation done once; O(N^2 * n_ensemble) sampling -- fine for N=600.

Reference: docs/endoderm_migration_models.md
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import cholesky, toeplitz

from simulation.params import SimParams
from simulation.analysis import orientation_acf
from simulation.theory import active_msd, active_msd_exponent

_HERE = Path(__file__).parent

# ── Physical parameters (Sde Boker PDF Jan 2026, same as hetero_abp.py) ───────
RADIUS_UM  = 6.5
T_K        = 310.0
ETA_PA_S   = 1e-3
V_BIO      = 0.0083    # µm/s = 30 µm/h
TAU_R_MEAN = 5400.0    # s = 90 min (biological persistence)

_phys   = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
D_T     = _phys.D_T
D_R_BIO = 1.0 / TAU_R_MEAN

# ── Simulation parameters ──────────────────────────────────────────────────────
DT_SIM     = 60.0   # s
N_STEPS    = 600    # 10 h
N_ENSEMBLE = 100    # trajectories per H value

# ── H values to compare ────────────────────────────────────────────────────────
# H=0.5  standard (white noise, reference)
# H=0.7  target for MSD ~ t^1.4
H_VALUES = [0.5, 0.6, 0.7, 0.8]

# D_H normalisation: per-step noise variance is the SAME for all H.
# This isolates the correlation effect from the amplitude effect.
#   2*D_H*dt^{2H} = 2*D_T*dt  =>  D_H = D_T * dt^{1-2H}
# At H=0.5: D_H = D_T  (standard).
# At H=0.7: D_H = D_T * 60^{-0.4} ~ 0.193*D_T (smaller coefficient, same step RMS).
def _D_H(H):
    return D_T * DT_SIM ** (1.0 - 2.0 * H)


# ── fGn generation ─────────────────────────────────────────────────────────────

def _fgn_cov(n, H):
    """Normalised autocovariance vector of fGn (Toeplitz row), length n.

    rho(0) = 1,  rho(k) = 0.5*((k+1)^{2H} - 2*k^{2H} + (k-1)^{2H})  k>=1
    """
    k = np.arange(n, dtype=float)
    rho = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                 + np.maximum(k - 1, 0) ** (2 * H))
    rho[0] = 1.0   # variance = 1 (formula gives 1 analytically, but set explicitly)
    return rho


def _cholesky_factor(n, H):
    """Lower-triangular Cholesky factor of the fGn covariance matrix."""
    rho = _fgn_cov(n, H)
    C   = toeplitz(rho)
    return cholesky(C, lower=True)


def sample_fgn(L, size, rng):
    """Draw `size` independent normalised fGn sequences of length n.

    Args:
        L:    (n, n) lower Cholesky factor of the fGn covariance.
        size: number of independent samples.
        rng:  numpy RNG.

    Returns:
        (size, n) array; each row has unit variance and the correct fGn correlations.
    """
    n = L.shape[0]
    Z = rng.standard_normal((n, size))
    return (L @ Z).T   # (size, n)


# ── Analytical MSD for Active FBM ─────────────────────────────────────────────

def active_fbm_msd_theory(t, D_H, H, D_R, v):
    """Exact ensemble-average MSD for Active FBM.

    MSD = 2*D_H*t^{2H}  +  (v^2/D_R)*(2t - 2*tau_R*(1-exp(-t/tau_R)))
    fBM and swim terms are independent so they add.
    Reduces to standard active_msd at H=0.5, D_H=D_T.
    """
    tau_R = 1.0 / D_R
    return (2.0 * D_H * t ** (2.0 * H)
            + (v ** 2 / D_R) * (2.0 * t - 2.0 * tau_R * (1.0 - np.exp(-t / tau_R))))


def active_fbm_msd_exponent(t, D_H, H, D_R, v):
    """Local log-log slope of Active FBM MSD: alpha(t) = d(log MSD)/d(log t)."""
    tau_R = 1.0 / D_R
    msd   = active_fbm_msd_theory(t, D_H, H, D_R, v)
    dmsd  = (2.0 * D_H * 2.0 * H * t ** (2.0 * H - 1.0)
             + (v ** 2 / D_R) * 2.0 * (1.0 - np.exp(-t / tau_R)))
    return t * dmsd / msd


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_active_fbm(H, rng, n_ensemble=N_ENSEMBLE):
    """Run n_ensemble Active FBM trajectories; return (trajs, phis).

    trajs: (n_ensemble, N_STEPS+1, 2) unwrapped positions.
    phis:  (N_STEPS+1, n_ensemble) orientations for DACF.
    """
    D_h = _D_H(H)
    dt  = DT_SIM
    n   = N_STEPS

    # Pre-generate all fGn noise (O(n^3) Cholesky once, then sample)
    L       = _cholesky_factor(n, H)
    # Scale: each fGn unit sample -> physical displacement
    scale   = np.sqrt(2.0 * D_h * dt ** (2.0 * H))
    xi_x    = sample_fgn(L, n_ensemble, rng) * scale   # (n_ens, n)
    xi_y    = sample_fgn(L, n_ensemble, rng) * scale   # (n_ens, n)

    # Angular noise (standard white noise, unchanged)
    eta_phi = rng.standard_normal((n, n_ensemble)) * np.sqrt(2.0 * D_R_BIO * dt)

    # Initialise
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
        phi += eta_phi[t]
        pos[t + 1, :, 0] = x
        pos[t + 1, :, 1] = y
        phis[t + 1]       = phi

    return pos, phis


def compute_msd(pos):
    """Ensemble-average MSD from (n_steps+1, N, 2) positions."""
    dx = pos[1:, :, 0] - pos[0:1, :, 0]
    dy = pos[1:, :, 1] - pos[0:1, :, 1]
    return (dx ** 2 + dy ** 2).mean(axis=1)


def compute_dacf(phis, max_lag):
    """Ensemble-average orientation ACF from (n_steps+1, N) phi array."""
    trajs = [
        np.column_stack([np.zeros((phis.shape[0], 2)), phis[:, i]])
        for i in range(phis.shape[1])
    ]
    _, acf = orientation_acf(trajs, max_lag=max_lag)
    return acf


def global_alpha(t, msd):
    return float(np.polyfit(np.log(t), np.log(msd), 1)[0])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Pe  = V_BIO ** 2 / (4.0 * D_T * D_R_BIO)
    print("=== Active FBM simulation ===")
    print(f"D_T={D_T:.4f} µm^2/s  |  D_R={D_R_BIO:.2e} rad^2/s  |  "
          f"V={V_BIO*3600:.0f} µm/h  |  Pe={Pe:.2f}")
    print(f"tau_R={TAU_R_MEAN/60:.0f} min  |  dt={DT_SIM:.0f} s  |  "
          f"N_steps={N_STEPS}  ({N_STEPS*DT_SIM/3600:.0f} h)  |  N_ens={N_ENSEMBLE}")
    print(f"Normalisation: D_H = D_T * dt^{{1-2H}} (same per-step noise amplitude for all H)")
    print()

    rng   = np.random.default_rng(seed=42)
    t_arr = np.arange(1, N_STEPS + 1) * DT_SIM
    t_h   = t_arr / 3600.0
    max_lag = N_STEPS // 2
    lag_h   = np.arange(max_lag) * DT_SIM / 3600.0

    # Colours and reference data
    colors = {0.5: "tab:blue", 0.6: "tab:green", 0.7: "tab:red", 0.8: "tab:purple"}

    results = {}
    for H in H_VALUES:
        D_h = _D_H(H)
        print(f"H={H}  D_H={D_h:.4f} µm^2/s^{{2H}}  (generating fGn + simulating)...")
        pos, phis = run_active_fbm(H, rng)
        msd   = compute_msd(pos)
        alpha = global_alpha(t_arr, msd)
        dacf  = compute_dacf(phis, max_lag)
        msd_th = active_fbm_msd_theory(t_arr, D_h, H, D_R_BIO, V_BIO)
        print(f"  global alpha = {alpha:.4f}  (theory peak alpha = "
              f"{active_fbm_msd_exponent(t_arr, D_h, H, D_R_BIO, V_BIO).max():.4f})")
        results[H] = {"msd": msd, "dacf": dacf, "alpha": alpha, "msd_th": msd_th}

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    ax_msd, ax_exp, ax_dacf = axes

    # ── Panel 1: MSD ─────────────────────────────────────────────────────────
    for H in H_VALUES:
        r  = results[H]
        lw = 2.5 if H in (0.5, 0.7) else 1.5
        ax_msd.loglog(t_h, r["msd"],   color=colors[H], lw=lw,
                      label=f"Active FBM  H={H}  (sim, alpha={r['alpha']:.3f})")
        ax_msd.loglog(t_h, r["msd_th"], color=colors[H], lw=lw * 0.6, ls=":",
                      alpha=0.7)  # theory dotted; one legend entry per H

    # Standard ABP theory (H=0.5, same as our biological-param ABP)
    msd_abp = active_msd(t_arr, D_T, D_R_BIO, V_BIO)
    ax_msd.loglog(t_h, msd_abp, color="black", lw=1.5, ls="--",
                  label=f"Standard ABP theory  (H=0.5 limit, alpha={global_alpha(t_arr, msd_abp):.3f})")

    # PDF reference lines (published prefactors, t in hours)
    t_ref = np.linspace(0.1, 10, 500)
    ax_msd.loglog(t_ref, 125.44 * t_ref ** 1.29,
                  color="gray", lw=1.2, ls="-.", alpha=0.8,
                  label="PDF 3D: 125.44 * t^1.29")
    ax_msd.loglog(t_ref, 193.92 * t_ref ** 1.40,
                  color="dimgray", lw=1.2, ls="--", alpha=0.8,
                  label="PDF 2D: 193.92 * t^1.40")

    ax_msd.set_xlabel("time [h]")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("MSD — Active FBM for varying H\n"
                     "Dotted = exact theory; solid = simulation")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 2: local exponent alpha(t) ─────────────────────────────────────
    for H in H_VALUES:
        D_h    = _D_H(H)
        alpha_t = active_fbm_msd_exponent(t_arr, D_h, H, D_R_BIO, V_BIO)
        lw = 2.5 if H in (0.5, 0.7) else 1.5
        ax_exp.semilogx(t_h, alpha_t, color=colors[H], lw=lw,
                        label=f"H={H}  (long-time -> {2*H:.1f})")
    ax_exp.axhline(1.40, color="dimgray", ls="--", lw=1.5, label="PDF 2D: alpha=1.40")
    ax_exp.axhline(1.29, color="gray",    ls="-.", lw=1.5, label="PDF 3D: alpha=1.29")
    ax_exp.axhline(1.0,  color="gray",    ls=":",  lw=1, alpha=0.5)
    ax_exp.axhline(2.0,  color="gray",    ls="--", lw=1, alpha=0.4,
                   label="alpha=2 (ballistic)")
    tau_h = TAU_R_MEAN / 3600.0
    ax_exp.axvline(tau_h, color="black", ls=":", lw=1, alpha=0.5,
                   label=f"tau_R = {TAU_R_MEAN/60:.0f} min")
    ax_exp.set_xlabel("time [h]")
    ax_exp.set_ylabel("alpha(t) = d(log MSD) / d(log t)")
    ax_exp.set_title("Local MSD exponent (theory)\n"
                     "Converges to 2H at long times")
    ax_exp.set_ylim(0.8, 2.2)
    ax_exp.legend(fontsize=8)
    ax_exp.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 3: DACF ─────────────────────────────────────────────────────────
    # All H values should give ~exponential DACF (angular process unchanged)
    dacf_th = np.exp(-D_R_BIO * np.arange(max_lag) * DT_SIM)
    ax_dacf.loglog(lag_h[1:], dacf_th[1:], color="black", lw=1.5, ls=":",
                   label=f"Theory: exp(-t/tau_R), tau_R={TAU_R_MEAN/60:.0f} min")
    for H in H_VALUES:
        acf = np.maximum(results[H]["dacf"][1:], 1e-6)
        lw = 2.5 if H in (0.5, 0.7) else 1.5
        ax_dacf.loglog(lag_h[1:], acf, color=colors[H], lw=lw,
                       label=f"H={H}")
    # PDF power-law reference
    anchor_idx = np.argmin(np.abs(lag_h[1:] - 0.5))
    acf_ref    = np.maximum(results[0.5]["dacf"][1:][anchor_idx], 1e-6)
    A_pl       = acf_ref / (0.5 ** (-0.83))
    t_pl       = lag_h[(lag_h > 0.2) & (lag_h < 5.0)]
    ax_dacf.loglog(t_pl, A_pl * t_pl ** (-0.83),
                   color="dimgray", lw=1.5, ls="--", alpha=0.8,
                   label="PDF measured: ~ t^-0.83 (2D, anchored)")
    ax_dacf.set_xlabel("lag time [h]")
    ax_dacf.set_ylabel("DACF  C_phi(tau)")
    ax_dacf.set_title("Orientation ACF — all H give exponential decay\n"
                      "(angular process is standard; fGn only in translation)")
    ax_dacf.legend(fontsize=8)
    ax_dacf.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle(
        f"Active FBM: fractional translational noise + standard ABP angular diffusion\n"
        f"v={V_BIO*3600:.0f} µm/h, tau_R={TAU_R_MEAN/60:.0f} min, "
        f"Pe={Pe:.2f}, normalised D_H=D_T*dt^{{1-2H}} (same per-step noise amplitude)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(_HERE / "active_fbm.png", dpi=150)
    plt.close(fig)
    print("\nSaved active_fbm.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'H':<8} {'D_H (µm^2/s^2H)':<22} {'global alpha':>14}  "
          f"{'long-time limit':>16}  {'vs 1.40':>10}")
    print("-" * 76)
    for H in H_VALUES:
        D_h   = _D_H(H)
        alpha = results[H]["alpha"]
        print(f"{H:<8.1f} {D_h:<22.5f} {alpha:>14.4f}  {2*H:>16.1f}  {alpha-1.40:>+10.4f}")


if __name__ == "__main__":
    main()
