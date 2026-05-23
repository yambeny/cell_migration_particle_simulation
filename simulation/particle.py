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

    def _apply_boundary(
        self, x_new: float, y_new: float, phi: float
    ) -> tuple[float, float, float]:
        """Apply the boundary condition from params and return corrected (x, y, phi)."""
        p = self.params
        if p.boundary == "none" or p.box_size is None:
            return x_new, y_new, phi

        L = p.box_size

        if p.boundary == "reflect":
            x_hit = False
            y_hit = False
            if x_new > L:
                x_new = 2.0 * L - x_new
                x_hit = True
            elif x_new < -L:
                x_new = -2.0 * L - x_new
                x_hit = True
            if y_new > L:
                y_new = 2.0 * L - y_new
                y_hit = True
            elif y_new < -L:
                y_new = -2.0 * L - y_new
                y_hit = True
            if x_hit:
                phi = np.pi - phi
            if y_hit:
                phi = -phi
            return x_new, y_new, phi

        if p.boundary == "stop":
            return float(np.clip(x_new, -L, L)), float(np.clip(y_new, -L, L)), phi

        if p.boundary == "slip":
            if x_new > L or x_new < -L:
                x_new = self.x   # block x: stay at current x
            if y_new > L or y_new < -L:
                y_new = self.y   # block y: stay at current y
            return x_new, y_new, phi

        return x_new, y_new, phi


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
        phi_new = self.phi + noise_r * eta[2]
        x_new = self.x + noise_t * eta[0]
        y_new = self.y + noise_t * eta[1]
        self.x, self.y, self.phi = self._apply_boundary(x_new, y_new, phi_new)


class ActiveBrownianParticle(Particle):
    """Self-propelled active Brownian particle — Eq. 4.

    Euler-Maruyama:
        x += v*cos(phi)*dt + sqrt(2*D_T*dt) * eta_x
        y += v*sin(phi)*dt + sqrt(2*D_T*dt) * eta_y
        phi += sqrt(2*D_R*dt) * eta_phi
    """

    def step(self) -> None:
        p = self.params
        eta = self._rng.standard_normal(3)
        noise_t = np.sqrt(2.0 * p.D_T * p.dt)
        noise_r = np.sqrt(2.0 * p.D_R * p.dt)
        phi_new = self.phi + noise_r * eta[2]
        x_new = self.x + p.v * np.cos(self.phi) * p.dt + noise_t * eta[0]
        y_new = self.y + p.v * np.sin(self.phi) * p.dt + noise_t * eta[1]
        self.x, self.y, self.phi = self._apply_boundary(x_new, y_new, phi_new)
