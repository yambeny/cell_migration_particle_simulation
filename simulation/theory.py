"""Analytical MSD predictions for passive and active Brownian particles in 2D.

All formulas are ensemble-averaged results for particles starting from a fixed
origin with a fixed initial orientation. Reference: Romanczuk et al. (2012),
Active Brownian Particles, EPJ Special Topics.
"""
import numpy as np


def passive_msd(t: np.ndarray, D_T: float) -> np.ndarray:
    """Exact MSD for passive Brownian motion in 2D: MSD = 4·D_T·t."""
    return 4.0 * D_T * t


def active_msd(t: np.ndarray, D_T: float, D_R: float, v: float) -> np.ndarray:
    """Exact analytical MSD for an active Brownian particle in 2D.

    MSD(t) = 4·D_T·t + (2v²/D_R)·[ t − (1 − exp(−D_R·t)) / D_R ]

    Reduces to passive_msd when v=0.
    Short-time limit (t ≪ τ_R): MSD ≈ 4·D_T·t + v²·t²  (ballistic, slope 2)
    Long-time limit  (t ≫ τ_R): MSD ≈ 4·D_eff·t         (diffusive, slope 1)
    """
    tau_R = 1.0 / D_R
    return 4.0 * D_T * t + (2.0 * v**2 / D_R) * (t - tau_R * (1.0 - np.exp(-t / tau_R)))


def active_msd_short_time(t: np.ndarray, D_T: float, v: float) -> np.ndarray:
    """Short-time (ballistic) asymptote: MSD ≈ 4·D_T·t + v²·t².

    Dominates for t ≪ τ_R = 1/D_R. The v²t² term gives slope 2 on a log-log
    plot when v >> sqrt(D_T/t).
    """
    return 4.0 * D_T * t + v**2 * t**2


def active_msd_long_time(t: np.ndarray, D_T: float, D_R: float, v: float) -> np.ndarray:
    """Long-time (enhanced diffusion) asymptote: MSD ≈ 4·D_eff·t.

    D_eff = D_T + v²/(2·D_R) is the swim-enhanced effective diffusion coefficient.
    Slope 1 on log-log, but with a larger prefactor than passive.
    """
    D_eff = D_T + v**2 / (2.0 * D_R)
    return 4.0 * D_eff * t


def effective_diffusion(D_T: float, D_R: float, v: float) -> float:
    """Swim-enhanced effective diffusion coefficient D_eff = D_T + v²/(2·D_R)."""
    return D_T + v**2 / (2.0 * D_R)


def rotational_relaxation_time(D_R: float) -> float:
    """Rotational relaxation time τ_R = 1/D_R [s]. Marks the ballistic→diffusive crossover."""
    return 1.0 / D_R


def active_msd_exponent(t: np.ndarray, D_T: float, D_R: float, v: float) -> np.ndarray:
    """Local log-log slope of active MSD: α(t) = d(log MSD)/d(log t) = t·dMSD/dt / MSD.

    α=1 at t→0 (diffusive), peaks slightly above 1 near t~τ_R when Pe>0, returns to 1 at t→∞.
    Only shows a clear ballistic peak (α→2) when τ_T = 4D_T/v² ≪ τ_R = 1/D_R (high Pe).
    """
    tau_R = 1.0 / D_R
    msd = active_msd(t, D_T, D_R, v)
    dmsd_dt = 4.0 * D_T + (2.0 * v**2 / D_R) * (1.0 - np.exp(-t / tau_R))
    return t * dmsd_dt / msd


def orientation_acf_theory(t: np.ndarray, D_R: float) -> np.ndarray:
    """Theoretical orientation ACF for ABP: exp(-D_R * t)."""
    return np.exp(-D_R * t)


def velocity_acf_theory(t: np.ndarray, v: float, D_R: float) -> np.ndarray:
    """Theoretical velocity ACF for ABP (lag > 0): v² * exp(-D_R * t)."""
    return v**2 * np.exp(-D_R * t)
