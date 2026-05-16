from abc import ABC, abstractmethod
import numpy as np
from simulation.params import SimParams


class Particle(ABC):
    def __init__(self, params: SimParams):
        self.params = params
        self.x = params.x0
        self.y = params.y0
        self.phi = params.phi0
        self._rng = np.random.default_rng(params.seed)

    @abstractmethod
    def step(self) -> None:
        """Advance particle state by one time step dt."""

    def state(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.phi)


class PassiveBrownianParticle(Particle):
    """Passive Brownian motion — Eq. 3.

    Euler-Maruyama:
        x += sqrt(2*D_T*dt) * eta_x
        y += sqrt(2*D_T*dt) * eta_y
        phi += sqrt(2*D_R*dt) * eta_phi
    """

    def step(self) -> None:
        p = self.params
        eta = self._rng.standard_normal(3)
        noise_t = np.sqrt(2.0 * p.D_T * p.dt)
        noise_r = np.sqrt(2.0 * p.D_R * p.dt)
        self.x += noise_t * eta[0]
        self.y += noise_t * eta[1]
        self.phi += noise_r * eta[2]
