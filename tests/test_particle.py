import numpy as np
import pytest
from simulation.params import SimParams


def test_simparams_defaults():
    p = SimParams()
    assert p.D_T == 0.22
    assert p.D_R == 0.16
    assert p.v == 0.0
    assert p.dt == 0.01
    assert p.n_steps == 1000
    assert p.seed == 42
