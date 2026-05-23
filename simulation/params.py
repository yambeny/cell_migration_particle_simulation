from __future__ import annotations
import math
from dataclasses import dataclass

_KB = 1.380649e-23  # Boltzmann constant [J/K]
_VALID_BOUNDARIES = {"none", "reflect", "stop", "slip"}


@dataclass
class SimParams:
    D_T: float = 0.22       # translational diffusion coefficient [µm²/s]
    D_R: float = 0.16       # rotational diffusion coefficient [rad²/s]
    v: float = 0.0          # self-propulsion speed [µm/s]; 0 = passive
    dt: float = 0.01        # time step [s]
    n_steps: int = 1000     # number of simulation steps
    seed: int = 42          # RNG seed; set to None for non-reproducible runs
    x0: float = 0.0         # initial x position [µm]
    y0: float = 0.0         # initial y position [µm]
    phi0: float = 0.0       # initial orientation angle [rad]
    boundary: str = "none"  # one of: "none", "reflect", "stop", "slip"
    box_size: float | None = None  # half-width L [µm]; box = [−L, L] × [−L, L]

    def __post_init__(self) -> None:
        if self.boundary not in _VALID_BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {_VALID_BOUNDARIES}, got {self.boundary!r}"
            )
        if self.boundary != "none" and self.box_size is None:
            raise ValueError("box_size must be set when boundary != 'none'")

    @classmethod
    def from_physical(
        cls,
        radius_um: float,
        T_K: float = 300.0,
        eta_Pa_s: float = 1e-3,
        **kwargs,
    ) -> SimParams:
        """Build SimParams with D_T and D_R derived from Stokes-Einstein-Debye.

        D_T = k_B T / (6π η R)   [µm²/s]   — translational (Eq. 1)
        D_R = k_B T / (8π η R³)  [rad²/s]  — rotational    (Eq. 2)

        Args:
            radius_um:  particle radius [µm]
            T_K:        temperature [K] (default 300 K)
            eta_Pa_s:   dynamic viscosity [Pa·s] (default 1e-3, water at ~20 °C)
            **kwargs:   any other SimParams fields (v, dt, n_steps, seed, …)
        """
        R = radius_um * 1e-6  # convert µm → m
        D_T = _KB * T_K / (6.0 * math.pi * eta_Pa_s * R) * 1e12  # m²/s → µm²/s
        D_R = _KB * T_K / (8.0 * math.pi * eta_Pa_s * R**3)      # rad²/s
        return cls(D_T=D_T, D_R=D_R, **kwargs)
