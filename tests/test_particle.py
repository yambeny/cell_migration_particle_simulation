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


# ── Boundary condition tests ──────────────────────────────────────────────────
import numpy as np

def _make(boundary, box_size=5.0, x0=0.0, y0=0.0, phi0=0.0,
           D_T=0.0, D_R=0.0, v=0.0, n_steps=1, seed=0):
    params = SimParams(
        D_T=D_T, D_R=D_R, v=v, dt=1.0, n_steps=n_steps,
        boundary=boundary, box_size=box_size,
        x0=x0, y0=y0, phi0=phi0, seed=seed,
    )
    return PassiveBrownianParticle(params)


# reflect
def test_reflect_position_mirrors_across_wall():
    """x=4.9, proposed x_new=5.1 → reflected to 4.9."""
    p = _make("reflect")
    x, y, phi = p._apply_boundary(5.1, 0.0, 0.0)
    assert abs(x - 4.9) < 1e-10
    assert y == 0.0


def test_reflect_orientation_flips_on_x_wall():
    """Hitting x-wall: phi → pi - phi."""
    p = _make("reflect")
    _, _, phi_out = p._apply_boundary(5.5, 0.0, 0.3)
    assert abs(phi_out - (np.pi - 0.3)) < 1e-10


def test_reflect_orientation_flips_on_y_wall():
    """Hitting y-wall: phi → -phi."""
    p = _make("reflect")
    _, _, phi_out = p._apply_boundary(0.0, 5.5, 0.3)
    assert abs(phi_out - (-0.3)) < 1e-10


def test_reflect_stays_inside_box():
    """Long run with high diffusion must never leave the box."""
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="reflect", box_size=5.0, seed=1)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


# stop
def test_stop_clamps_to_wall():
    """Proposed x_new=7.0 with L=5 → x clamped to 5.0."""
    p = _make("stop")
    x, y, phi = p._apply_boundary(7.0, 0.0, 0.5)
    assert x == 5.0
    assert y == 0.0
    assert phi == 0.5  # phi unchanged


def test_stop_stays_inside_box():
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="stop", box_size=5.0, seed=2)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


# slip
def test_slip_blocks_crossing_axis_keeps_parallel():
    """At x0=4.9, x_new=5.4 (exits), y_new=0.5 (stays) → x blocked, y moves."""
    p = _make("slip", x0=4.9)
    x, y, phi = p._apply_boundary(5.4, 0.5, 0.3)
    assert abs(x - 4.9) < 1e-10   # x blocked: stays at x0
    assert abs(y - 0.5) < 1e-10   # y passes through
    assert phi == 0.3              # phi unchanged


def test_slip_stays_inside_box():
    params = SimParams(D_T=2.0, D_R=0.0, v=0.0, dt=0.1, n_steps=2000,
                       boundary="slip", box_size=5.0, seed=3)
    p = PassiveBrownianParticle(params)
    for _ in range(2000):
        p.step()
    assert -5.0 <= p.x <= 5.0
    assert -5.0 <= p.y <= 5.0


def test_boundary_none_is_unchanged():
    """boundary='none' must produce bit-identical results to old behavior."""
    params_old = SimParams(D_T=0.22, D_R=0.16, v=1.0, dt=0.01, n_steps=100, seed=99)
    params_new = SimParams(D_T=0.22, D_R=0.16, v=1.0, dt=0.01, n_steps=100, seed=99,
                           boundary="none")
    from simulation.particle import ActiveBrownianParticle
    from simulation.simulator import Simulator
    traj_old = Simulator(ActiveBrownianParticle(params_old), params_old).run()
    traj_new = Simulator(ActiveBrownianParticle(params_new), params_new).run()
    np.testing.assert_array_equal(traj_old, traj_new)
