"""ABP + CIL (Contact Inhibition of Locomotion) simulation.

N cells in a 2D periodic box. Three conditions compared:
  1. Non-interacting ABP  (K_REP=0,   K_CIL=0)   — baseline, matches ABP theory
  2. ABP + steric only    (K_REP>0,   K_CIL=0)   — pure crowding, no repolarisation
  3. ABP + steric + CIL   (K_REP>0,   K_CIL>0)   — full model

CIL rule: when |r_ij| < 2R, cell i receives an angular torque pushing phi_i
toward the direction AWAY from cell j:
  dphi_i += K_CIL * sum_{j in contact} sin(theta_away_ij - phi_i) * dt

All conditions use biological parameters from the Sde Boker PDF (Jan 2026).
MSD and DACF are ensemble-averaged over all N cells in the box.

Reference: docs/endoderm_migration_models.md
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from simulation.params import SimParams
from simulation.analysis import orientation_acf
from simulation.theory import passive_msd, active_msd, active_msd_exponent

_HERE = Path(__file__).parent

# ── Physical parameters (Sde Boker PDF Jan 2026) ──────────────────────────────
RADIUS_UM  = 6.5
T_K        = 310.0
ETA_PA_S   = 1e-3
V_BIO      = 0.0083    # µm/s = 30 µm/h
TAU_R_MEAN = 5400.0    # s = 90 min (biological persistence)

_phys   = SimParams.from_physical(radius_um=RADIUS_UM, T_K=T_K, eta_Pa_s=ETA_PA_S)
D_T     = _phys.D_T
D_R_BIO = 1.0 / TAU_R_MEAN

# ── Multi-cell parameters ──────────────────────────────────────────────────────
N_CELLS   = 100           # cells in the box
PACK_FRAC = 0.60          # area packing fraction
D_CONTACT = 2.0 * RADIUS_UM  # 13 µm — contact threshold

# Steric repulsion: K_REP [s^-1] × overlap [µm] → velocity [µm/s].
# K_REP=0.02 s^-1: at 1 µm overlap → 0.02 µm/s >> v=0.0083 µm/s; dt*K_REP=0.6 < 1 (stable)
K_REP = 0.02

# CIL rate: K_CIL [s^-1]. tau_CIL = 1/K_CIL = 600 s = 10 min.
# CIL repolarisation is faster than rotational diffusion (tau_R = 90 min).
K_CIL = 1.0 / 600.0

# ── Simulation timing ──────────────────────────────────────────────────────────
DT_SIM  = 30.0    # s — smaller than hetero_abp.py for steric stability
N_EQUIL = 200     # equilibration steps (100 min); discard from MSD
N_STEPS = 1200    # measurement steps (600 min = 10 h)

# Derived box size
L_FULL = float(np.sqrt(N_CELLS * np.pi * RADIUS_UM**2 / PACK_FRAC))

# Conditions to compare
CONDITIONS = [
    ("ABP  (no interaction)",  0.0,   0.0  ),
    ("ABP + steric only",      K_REP, 0.0  ),
    ("ABP + steric + CIL",     K_REP, K_CIL),
]
COLORS = {
    "ABP  (no interaction)": "tab:blue",
    "ABP + steric only":     "tab:orange",
    "ABP + steric + CIL":    "tab:red",
}


# ── Multi-cell simulation ──────────────────────────────────────────────────────

class CILSimulation:
    """N active Brownian particles in a 2D periodic box.

    Translational update (overdamped):
        dx_i = v cos(phi_i) dt + k_rep * sum_j overlap_ij * rhat_ij dt + sqrt(2 D_T dt) eta
    Rotational update:
        dphi_i = sqrt(2 D_R dt) eta + k_cil * sum_{j in contact} sin(theta_away_ij - phi_i) dt
    """

    def __init__(self, n, L, D_T, D_R, v, dt,
                 k_rep=0.0, k_cil=0.0, d_contact=13.0, seed=42):
        self.n = n
        self.L = L
        self.D_T = D_T
        self.D_R = D_R
        self.v = v
        self.dt = dt
        self.k_rep = k_rep
        self.k_cil = k_cil
        self.d_contact = d_contact
        self._rng = np.random.default_rng(seed)
        self._init_positions()

    def _init_positions(self):
        """Grid placement with small perturbation; guarantees no initial overlap
        when grid spacing > 2R (holds for PACK_FRAC < pi/4 ~ 0.785)."""
        n_side = int(np.ceil(np.sqrt(self.n)))
        spacing = self.L / n_side
        idx = np.arange(self.n)
        col = idx % n_side
        row = idx // n_side
        jitter = 0.05 * spacing
        self.x = ((col + 0.5) * spacing
                  + self._rng.uniform(-jitter, jitter, self.n)) % self.L
        self.y = ((row + 0.5) * spacing
                  + self._rng.uniform(-jitter, jitter, self.n)) % self.L
        self.phi = self._rng.uniform(0.0, 2.0 * np.pi, self.n)
        # Unwrapped positions for MSD (reset after equilibration via reset_unwrapped)
        self.x_unwrap = self.x.copy()
        self.y_unwrap = self.y.copy()

    def reset_unwrapped(self):
        """Anchor MSD measurement to current (equilibrated) positions."""
        self.x_unwrap = self.x.copy()
        self.y_unwrap = self.y.copy()

    def _min_image(self, dx, dy):
        L = self.L
        return dx - L * np.round(dx / L), dy - L * np.round(dy / L)

    def step(self):
        dt = self.dt
        steric_x = steric_y = cil_torque = 0.0

        if self.k_rep > 0 or self.k_cil > 0:
            # Pairwise separation (N,N) with minimum image convention
            dx_ij = self.x[:, None] - self.x[None, :]   # x_i - x_j
            dy_ij = self.y[:, None] - self.y[None, :]
            dx_ij, dy_ij = self._min_image(dx_ij, dy_ij)

            # Distance; diagonal = inf (exclude self)
            dist_sq = dx_ij ** 2 + dy_ij ** 2
            np.fill_diagonal(dist_sq, np.inf)
            dist_ij = np.sqrt(dist_sq)          # inf on diagonal
            inv_d   = 1.0 / dist_ij             # 0 on diagonal (1/inf)

            if self.k_rep > 0:
                overlap  = np.maximum(0.0, self.d_contact - dist_ij)  # (N,N)
                steric_x = (overlap * dx_ij * inv_d).sum(axis=1) * self.k_rep
                steric_y = (overlap * dy_ij * inv_d).sum(axis=1) * self.k_rep

            if self.k_cil > 0:
                in_contact  = dist_ij < self.d_contact          # (N,N) bool
                theta_away  = np.arctan2(dy_ij, dx_ij)          # direction i away from j
                phi_row     = self.phi[:, None]                  # (N,1) broadcast to (N,N)
                cil_torque  = (in_contact
                               * np.sin(theta_away - phi_row)
                               ).sum(axis=1) * self.k_cil        # (N,)

        # Euler-Maruyama update
        noise_t = np.sqrt(2.0 * self.D_T * dt)
        noise_r = np.sqrt(2.0 * self.D_R * dt)
        eta = self._rng.standard_normal((3, self.n))

        dx = self.v * np.cos(self.phi) * dt + steric_x * dt + noise_t * eta[0]
        dy = self.v * np.sin(self.phi) * dt + steric_y * dt + noise_t * eta[1]
        dp = noise_r * eta[2] + cil_torque * dt

        self.x = (self.x + dx) % self.L
        self.y = (self.y + dy) % self.L
        self.phi += dp
        self.x_unwrap += dx
        self.y_unwrap += dy

    def run(self, n_steps):
        """Run n_steps; return (positions (n+1,N,2), phis (n+1,N))."""
        pos  = np.empty((n_steps + 1, self.n, 2))
        phis = np.empty((n_steps + 1, self.n))
        pos[0, :, 0] = self.x_unwrap
        pos[0, :, 1] = self.y_unwrap
        phis[0]      = self.phi.copy()
        for s in range(1, n_steps + 1):
            self.step()
            pos[s, :, 0] = self.x_unwrap
            pos[s, :, 1] = self.y_unwrap
            phis[s]      = self.phi.copy()
        return pos, phis


# ── Analysis helpers ───────────────────────────────────────────────────────────

def ensemble_msd(positions):
    """Ensemble-average MSD from (n_steps+1, N, 2) unwrapped positions."""
    dx = positions[1:, :, 0] - positions[0:1, :, 0]
    dy = positions[1:, :, 1] - positions[0:1, :, 1]
    return (dx ** 2 + dy ** 2).mean(axis=1)


def ensemble_dacf(phis, max_lag):
    """Ensemble-average orientation ACF from (n_steps+1, N) phi array."""
    trajs = [
        np.column_stack([np.zeros((phis.shape[0], 2)), phis[:, i]])
        for i in range(phis.shape[1])
    ]
    _, acf = orientation_acf(trajs, max_lag=max_lag)
    return acf


def global_alpha(t, msd):
    return float(np.polyfit(np.log(t), np.log(msd), 1)[0])


def smoothed_local_alpha(t_arr, msd, window=40):
    """d(log MSD)/d(log t) with uniform smoothing to suppress noise."""
    log_msd = uniform_filter1d(np.log(msd), size=window)
    log_t   = np.log(t_arr)
    return np.gradient(log_msd, log_t)


def run_condition(label, k_rep, k_cil, seed=42):
    """Equilibrate, reset, measure.  Returns dict of results."""
    print(f"  [{label}]")
    sim = CILSimulation(N_CELLS, L_FULL, D_T, D_R_BIO, V_BIO, DT_SIM,
                        k_rep=k_rep, k_cil=k_cil,
                        d_contact=D_CONTACT, seed=seed)
    print(f"    Equilibrating {N_EQUIL} steps ({N_EQUIL*DT_SIM/60:.0f} min)...")
    for _ in range(N_EQUIL):
        sim.step()

    snap_x, snap_y = sim.x.copy(), sim.y.copy()
    sim.reset_unwrapped()

    print(f"    Measuring {N_STEPS} steps ({N_STEPS*DT_SIM/3600:.1f} h)...")
    positions, phis = sim.run(N_STEPS)

    t_arr = np.arange(1, N_STEPS + 1) * DT_SIM
    msd   = ensemble_msd(positions)
    alpha = global_alpha(t_arr, msd)
    dacf  = ensemble_dacf(phis, max_lag=N_STEPS // 2)
    alpha_t = smoothed_local_alpha(t_arr, msd)

    print(f"    global alpha = {alpha:.4f}")
    return {
        "msd": msd, "dacf": dacf, "alpha": alpha,
        "alpha_t": alpha_t, "snap": (snap_x, snap_y),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Pe  = V_BIO ** 2 / (4.0 * D_T * D_R_BIO)
    tau_T = 4.0 * D_T / V_BIO ** 2
    print("=== ABP + CIL simulation ===")
    print(f"D_T={D_T:.4f} µm^2/s  |  D_R_bio={D_R_BIO:.2e} rad^2/s  |  "
          f"V={V_BIO*3600:.0f} µm/h  |  Pe={Pe:.2f}")
    print(f"tau_R={TAU_R_MEAN/60:.0f} min  |  tau_T={tau_T/60:.0f} min  "
          f"({'ballistic regime EXISTS' if tau_T < TAU_R_MEAN else 'NO ballistic regime'})")
    print(f"N={N_CELLS} cells  |  phi={PACK_FRAC}  |  L={L_FULL:.1f} µm  "
          f"|  tau_CIL={1/K_CIL/60:.0f} min  |  dt={DT_SIM:.0f} s")
    print()

    t_arr = np.arange(1, N_STEPS + 1) * DT_SIM
    t_h   = t_arr / 3600.0
    max_lag = N_STEPS // 2
    lag_h   = np.arange(max_lag) * DT_SIM / 3600.0

    # ── Run conditions ────────────────────────────────────────────────────────
    print("Running conditions (same seed -- controlled comparison):")
    results = {}
    for label, k_rep, k_cil in CONDITIONS:
        results[label] = run_condition(label, k_rep, k_cil, seed=42)

    # ── Analytical references ─────────────────────────────────────────────────
    msd_passive_th = passive_msd(t_arr, D_T)
    msd_abp_th     = active_msd(t_arr, D_T, D_R_BIO, V_BIO)
    alpha_passive_th = np.ones_like(t_arr)
    alpha_abp_th     = active_msd_exponent(t_arr, D_T, D_R_BIO, V_BIO)
    alpha_passive_g  = global_alpha(t_arr, msd_passive_th)
    alpha_abp_g      = global_alpha(t_arr, msd_abp_th)
    dacf_abp_th      = np.exp(-D_R_BIO * np.arange(max_lag) * DT_SIM)

    # ── Figure: 2×2 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_msd, ax_alpha = axes[0]
    ax_dacf, ax_snap = axes[1]

    # ── Panel 1: MSD ──────────────────────────────────────────────────────────
    ax_msd.loglog(t_h, msd_passive_th, color="tab:gray", lw=1.5, ls=":",
                  label=f"Passive theory  (alpha={alpha_passive_g:.3f})")
    ax_msd.loglog(t_h, msd_abp_th,     color="tab:blue", lw=1.5, ls=":",
                  label=f"ABP theory (no CIL, single cell)  (alpha={alpha_abp_g:.3f})")
    for label, _, _ in CONDITIONS:
        r  = results[label]
        lw = 2.5 if "CIL" in label else 1.8
        ls = "-" if "CIL" in label else "--"
        ax_msd.loglog(t_h, r["msd"], color=COLORS[label], lw=lw, ls=ls,
                      label=f"{label}  (alpha={r['alpha']:.3f})")
    t_ref = np.linspace(0.1, 10, 500)
    ax_msd.loglog(t_ref, 125.44 * t_ref ** 1.29,
                  color="black",   lw=1.2, ls="-.", alpha=0.7,
                  label="PDF measured: 3D  t^1.29")
    ax_msd.loglog(t_ref, 193.92 * t_ref ** 1.40,
                  color="dimgray", lw=1.2, ls="--", alpha=0.7,
                  label="PDF measured: 2D  t^1.40")
    ax_msd.set_xlabel("time [h]")
    ax_msd.set_ylabel("MSD [µm²]")
    ax_msd.set_title("MSD — passive / ABP / ABP+steric / ABP+CIL")
    ax_msd.legend(fontsize=7)
    ax_msd.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 2: local exponent alpha(t) ─────────────────────────────────────
    ax_alpha.semilogx(t_h, alpha_passive_th, color="tab:gray", lw=1.5, ls=":",
                      label="Passive theory  (alpha=1)")
    ax_alpha.semilogx(t_h, alpha_abp_th,     color="tab:blue", lw=1.5, ls=":",
                      label="ABP theory (single cell)")
    for label, _, _ in CONDITIONS[1:]:    # skip non-interacting (==ABP theory)
        r = results[label]
        ax_alpha.semilogx(t_h, r["alpha_t"], color=COLORS[label], lw=2,
                          label=f"{label}  (global={r['alpha']:.3f}, smoothed)")
    ax_alpha.axhline(1.40, color="dimgray", ls="--", lw=1.5, label="PDF 2D: alpha=1.40")
    ax_alpha.axhline(1.29, color="black",   ls="-.", lw=1.5, label="PDF 3D: alpha=1.29")
    ax_alpha.axhline(1.0,  color="gray",    ls=":",  lw=1, alpha=0.5)
    ax_alpha.axhline(2.0,  color="gray",    ls="--", lw=1, alpha=0.4,
                     label="alpha=2 (ballistic)")
    ax_alpha.set_xlabel("time [h]")
    ax_alpha.set_ylabel("alpha(t) = d(log MSD)/d(log t)")
    ax_alpha.set_title("Local MSD exponent  alpha(t)\n"
                       "(CIL/steric curves smoothed to suppress ensemble noise)")
    ax_alpha.set_ylim(0.5, 2.5)
    ax_alpha.legend(fontsize=7.5)
    ax_alpha.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 3: DACF log-log ─────────────────────────────────────────────────
    ax_dacf.loglog(lag_h[1:], dacf_abp_th[1:], color="tab:blue", lw=1.5, ls=":",
                   alpha=0.7, label=f"ABP theory: exp(-t/tau_R), tau_R={TAU_R_MEAN/60:.0f} min")
    for label, _, _ in CONDITIONS:
        acf = np.maximum(results[label]["dacf"][1:], 1e-6)
        lw  = 2.5 if "CIL" in label else 1.8
        ls  = "-" if "CIL" in label else "--"
        ax_dacf.loglog(lag_h[1:], acf, color=COLORS[label], lw=lw, ls=ls,
                       label=label)
    # PDF power-law reference, anchored to no-interaction DACF at lag=0.5 h
    acf_ref = np.maximum(results["ABP  (no interaction)"]["dacf"][1:], 1e-6)
    anchor  = np.argmin(np.abs(lag_h[1:] - 0.5))
    A_pl    = float(acf_ref[anchor]) / (0.5 ** (-0.83))
    t_pl    = lag_h[(lag_h > 0.2) & (lag_h < 5.0)]
    ax_dacf.loglog(t_pl, A_pl * t_pl ** (-0.83),
                   color="black", lw=1.5, ls="--", alpha=0.7,
                   label="PDF measured: ~ t^-0.83  (2D, anchored)")
    ax_dacf.set_xlabel("lag time [h]")
    ax_dacf.set_ylabel("DACF  C_phi(tau)")
    ax_dacf.set_title("Orientation ACF — does CIL produce non-exponential decay?")
    ax_dacf.legend(fontsize=7.5)
    ax_dacf.grid(True, which="both", ls="--", alpha=0.4)

    # ── Panel 4: cell snapshot (ABP+CIL equilibrated configuration) ───────────
    snap_x, snap_y = results["ABP + steric + CIL"]["snap"]
    ax_snap.set_facecolor("#f8f8f0")
    for xi, yi in zip(snap_x, snap_y):
        c = plt.Circle((xi, yi), RADIUS_UM,
                        facecolor="tab:red", edgecolor="darkred",
                        alpha=0.55, linewidth=0.5)
        ax_snap.add_patch(c)
    ax_snap.set_xlim(0, L_FULL)
    ax_snap.set_ylim(0, L_FULL)
    ax_snap.set_aspect("equal")
    ax_snap.set_xlabel("x [µm]")
    ax_snap.set_ylabel("y [µm]")
    ax_snap.set_title(
        f"Cell positions after equilibration (ABP+CIL)\n"
        f"N={N_CELLS}, phi={PACK_FRAC}, L={L_FULL:.0f} µm, R={RADIUS_UM} µm"
    )

    fig.suptitle(
        f"ABP + CIL: endoderm migration  |  "
        f"v={V_BIO*3600:.0f} µm/h, tau_R={TAU_R_MEAN/60:.0f} min, Pe={Pe:.2f}, "
        f"phi={PACK_FRAC}, tau_CIL={1/K_CIL/60:.0f} min",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(_HERE / "abp_cil.png", dpi=150)
    plt.close(fig)
    print("\nSaved abp_cil.png")

    # ── Snapshot comparison: all 3 conditions side-by-side ────────────────────
    fig2, snap_axes = plt.subplots(1, 3, figsize=(15, 5))
    snap_colors = list(COLORS.values())
    for ax, (label, _, _), col in zip(snap_axes, CONDITIONS, snap_colors):
        sx, sy = results[label]["snap"]
        ax.set_facecolor("#f8f8f0")
        for xi, yi in zip(sx, sy):
            c = plt.Circle((xi, yi), RADIUS_UM,
                            facecolor=col, edgecolor="gray",
                            alpha=0.55, linewidth=0.5)
            ax.add_patch(c)
        ax.set_xlim(0, L_FULL)
        ax.set_ylim(0, L_FULL)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nalpha={results[label]['alpha']:.3f}", fontsize=9)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
    fig2.suptitle(
        f"Equilibrated cell configurations (phi={PACK_FRAC}, N={N_CELLS})\n"
        f"All conditions start from identical grid positions (seed=42)",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(_HERE / "abp_cil_snapshots.png", dpi=150)
    plt.close(fig2)
    print("Saved abp_cil_snapshots.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Condition':<30}  {'alpha':>8}  {'vs 1.40':>10}  {'vs 1.29':>10}")
    print("-" * 65)
    for lbl, alpha in [("Passive theory",  alpha_passive_g),
                       ("ABP theory",      alpha_abp_g)] + \
                      [(l, results[l]["alpha"]) for l, *_ in CONDITIONS]:
        print(f"{lbl:<30}  {alpha:>8.4f}  {alpha-1.40:>+10.4f}  {alpha-1.29:>+10.4f}")

    # Interpretation
    cil_alpha = results["ABP + steric + CIL"]["alpha"]
    steric_alpha = results["ABP + steric only"]["alpha"]
    print(f"\nCrowding effect (steric - no interaction): "
          f"{steric_alpha - results['ABP  (no interaction)']['alpha']:+.4f}")
    print(f"CIL effect on top of steric:               "
          f"{cil_alpha - steric_alpha:+.4f}")
    print(f"Total collective effect (CIL+steric vs ABP theory): "
          f"{cil_alpha - alpha_abp_g:+.4f}")


if __name__ == "__main__":
    main()
