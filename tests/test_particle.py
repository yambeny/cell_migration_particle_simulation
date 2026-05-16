from simulation.params import SimParams
from simulation.particle import PassiveBrownianParticle, ActiveBrownianParticle


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


def test_active_particle_zero_noise_moves_straight():
    """With D_T=D_R=0, particle travels exactly v*dt in direction phi0."""
    params = SimParams(D_T=0.0, D_R=0.0, v=1.0, dt=0.1, phi0=0.0, seed=0)
    p = ActiveBrownianParticle(params)
    p.step()
    assert abs(p.x - 0.1) < 1e-10   # v*cos(0)*dt = 1.0*1.0*0.1
    assert abs(p.y - 0.0) < 1e-10   # v*sin(0)*dt = 0


def test_active_particle_v0_matches_passive():
    """Active particle with v=0 must produce identical trajectory to passive (same seed)."""
    kwargs = dict(D_T=0.22, D_R=0.16, v=0.0, dt=0.01, seed=7)
    # Two separate SimParams so each particle gets its own independent RNG seeded at 7
    passive = PassiveBrownianParticle(SimParams(**kwargs))
    active = ActiveBrownianParticle(SimParams(**kwargs))
    for _ in range(20):
        passive.step()
        active.step()
    assert abs(passive.x - active.x) < 1e-10
    assert abs(passive.y - active.y) < 1e-10
    assert abs(passive.phi - active.phi) < 1e-10


def test_active_particle_step_is_deterministic_with_seed():
    params = SimParams(v=2.0, seed=99)
    p1 = ActiveBrownianParticle(params)
    p2 = ActiveBrownianParticle(params)
    for _ in range(10):
        p1.step()
        p2.step()
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.phi == p2.phi
