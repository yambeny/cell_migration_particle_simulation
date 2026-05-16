import numpy as np
from simulation.particle import Particle
from simulation.params import SimParams


class Simulator:
    """Time-steps a particle and records the full trajectory.

    Returns an array of shape (n_steps+1, 3): rows are time snapshots,
    columns are [x, y, phi]. Row 0 is the initial state.

    Holds a list internally so multi-particle support can be added later
    without changing the interface.
    """

    def __init__(self, particle: Particle, params: SimParams):
        self.particle = particle
        self.params = params

    def run(self) -> np.ndarray:
        p = self.params
        traj = np.empty((p.n_steps + 1, 3))
        traj[0] = self.particle.state()
        for i in range(p.n_steps):
            self.particle.step()
            traj[i + 1] = self.particle.state()
        return traj
