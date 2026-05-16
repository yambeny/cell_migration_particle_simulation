from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle


def test_simparams_defaults():
    p = SimParams()
    assert p.D_T == 0.22
    assert p.D_R == 0.16
    assert p.v == 0.0
    assert p.dt == 0.01
    assert p.n_steps == 1000
    assert p.seed == 42
    assert p.x0 == 0.0
    assert p.y0 == 0.0
    assert p.phi0 == 0.0



def test_passive_particle_initializes():
    params = SimParams(x0=1.0, y0=2.0, phi0=0.5)
    p = PassiveBrownianParticle(params)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.phi == 0.5


def test_passive_particle_step_is_deterministic_with_seed():
    params = SimParams(seed=0)
    p1 = PassiveBrownianParticle(params)
    p2 = PassiveBrownianParticle(params)
    p1.step()
    p2.step()
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.phi == p2.phi


def test_passive_particle_zero_diffusion_stays_put():
    params = SimParams(D_T=0.0, D_R=0.0, seed=0)
    p = PassiveBrownianParticle(params)
    p.step()
    assert p.x == 0.0
    assert p.y == 0.0
    assert p.phi == 0.0
