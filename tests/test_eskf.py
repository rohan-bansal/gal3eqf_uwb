import numpy as np
import pytest

from gal3uwb import lie
from gal3uwb.filters.eskf import DIM, GAMMA, State, propagate, range_model
from gal3uwb.params import C

DT = 0.005


def _state():
    rng = np.random.default_rng(0)
    return State(
        lie.exp(np.array([0.2, -0.3, 0.5])),
        np.array([1.5, -0.7, 0.2]),
        np.array([1.0, 2.0, 1.0]),
        np.array([0.01, -0.02, 0.005]),
        np.array([0.05, 0.03, -0.04]),
        3.0e5 + rng.normal(),
        C * 2e-5,
    )


GYRO = np.array([0.4, -1.2, 0.8])
ACCEL = np.array([1.0, -0.5, 9.9])


def _numerical_F(x, coupled, steps):
    x1, _ = propagate(x, GYRO, ACCEL, DT, coupled)
    F = np.zeros((DIM, DIM))
    for i in range(DIM):
        d = np.zeros(DIM)
        d[i] = steps[i]
        xp, _ = propagate(x.retract(d), GYRO, ACCEL, DT, coupled)
        xm, _ = propagate(x.retract(-d), GYRO, ACCEL, DT, coupled)
        F[:, i] = (x1.local(xp) - x1.local(xm)) / (2 * steps[i])
    return F


@pytest.mark.parametrize("coupled", [False, True])
def test_propagation_jacobian(coupled):
    x = _state()
    steps = np.full(DIM, 1e-6)
    steps[GAMMA] = 1e3  # gamma enters through gamma / C
    _, F = propagate(x, GYRO, ACCEL, DT, coupled)
    F_num = _numerical_F(x, coupled, steps)
    other = [i for i in range(DIM) if i != GAMMA]
    # Bias columns drop O(dt^2) terms; everything else is exact to first order.
    np.testing.assert_allclose(F[:, other], F_num[:, other], atol=2e-5)
    np.testing.assert_allclose(F[:, GAMMA], F_num[:, GAMMA], rtol=2e-3, atol=1e-14)


def test_range_jacobian():
    x = _state()
    anchor = np.array([-3.0, 3.0, 2.8])
    _, H = range_model(x, anchor)
    H_num = np.zeros(DIM)
    step = 1e-3  # beta is ~3e5 m, so smaller steps lose precision
    for i in range(DIM):
        d = np.zeros(DIM)
        d[i] = step
        H_num[i] = (range_model(x.retract(d), anchor)[0] - range_model(x.retract(-d), anchor)[0]) / (2 * step)
    np.testing.assert_allclose(H, H_num, atol=1e-6)
