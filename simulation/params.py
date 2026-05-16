from dataclasses import dataclass


@dataclass
class SimParams:
    D_T: float = 0.22      # translational diffusion coefficient [µm²/s]
    D_R: float = 0.16      # rotational diffusion coefficient [rad²/s]
    v: float = 0.0         # self-propulsion speed [µm/s]; 0 = passive
    dt: float = 0.01       # time step [s]
    n_steps: int = 1000    # number of simulation steps
    seed: int = 42         # RNG seed; set to None for non-reproducible runs
    x0: float = 0.0        # initial x position [µm]
    y0: float = 0.0        # initial y position [µm]
    phi0: float = 0.0      # initial orientation angle [rad]
