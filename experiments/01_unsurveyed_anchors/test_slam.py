"""Checks of the SLAM filters: Jacobians against finite differences, and the
unobservable directions (rotating the scene about gravity, translating it).

    .venv/bin/python -m pytest experiments/01_unsurveyed_anchors/test_slam.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gal3uwb import gal3, lie  # noqa: E402
from gal3uwb.filters.common import Estimate, FilterNoise  # noqa: E402
from gal3uwb.filters.eqf import frame_from_estimate  # noqa: E402
from gal3uwb.params import ClockNoise, ImuNoise  # noqa: E402

from ekf_fej import EkfFej  # noqa: E402
from ekf_slam import EkfSlam  # noqa: E402
from eqf_slam import N_CORE, EqfSlam  # noqa: E402
from model import SlamEstimate, prior_covariance  # noqa: E402

GYRO, ACCEL, DT = np.array([0.3, -0.2, 0.5]), np.array([0.4, -0.3, 9.6]), 0.005
ANCHORS = np.array([[3.0, 3.0, 2.8], [-3.0, 3.1, 0.2], [3.2, -3.0, 0.3], [-2.9, -3.1, 2.7]])
LEVER = np.array([0.13, -0.17, -0.05])
NOISE = FilterNoise(ImuNoise(), ClockNoise(), 0.1)
SIGMA = {"rot": np.full(3, 0.05), "vel": 0.1, "pos": 0.1, "gyro_bias": 0.01, "accel_bias": 0.05, "delta": 3e-9,
         "rho": 2e-5}


def _state(n_clocks):
    nav = Estimate(lie.exp(np.array([0.1, -0.2, 0.7])), np.array([1.5, -0.8, 0.2]), np.array([1.2, 0.4, 1.1]),
                   np.array([0.01, -0.02, 0.005]), np.array([0.05, -0.03, 0.02]), 7e-4, 2e-5,
                   np.array([3e-4, -2e-4, 1e-4])[:n_clocks], np.array([1e-5, -2e-5, 3e-6])[:n_clocks])
    return SlamEstimate(nav, ANCHORS.copy())


def _filters(n_clocks):
    x = _state(n_clocks)
    P0 = prior_covariance(SIGMA, n_clocks, len(ANCHORS))
    eqf = EqfSlam(x, P0, NOISE)
    eqf.b[6:9] = [1e-3, -2e-3, 5e-4]  # a nonzero virtual bias exercises more terms
    return EkfSlam(x, P0, NOISE), eqf


def _numeric_jacobian(fun, n, step):
    cols = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = step[i] if np.ndim(step) else step
        cols.append((fun(d) - fun(-d)) / (2 * d[i]))
    return np.array(cols).T


def _perturbed(f, eps):
    g = copy.deepcopy(f)
    g.apply(eps)
    return g


def _eqf_error(f_true, f_hat):
    return f_hat.error_to(f_true.F, f_true.b, f_true.A, f_true.ac, f_true.ak)


def _eqf_steps(f):
    """Finite-difference steps: the time coordinate and anchor clocks are in seconds."""
    step = np.full(f.dim, 1e-5)
    step[9] = 1e-13
    step[N_CORE:f.ia] = 1e-13
    return step


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_eqf_range_and_sync_jacobians(n_clocks):
    _, f = _filters(n_clocks)
    step = _eqf_steps(f)
    _, H = f.range_model(LEVER)
    num = _numeric_jacobian(lambda e: _perturbed(f, e).range_model(LEVER)[0], f.dim, step)
    np.testing.assert_allclose(H, num, rtol=1e-5, atol=2e-5)
    for tx in range(len(ANCHORS)):
        _, H, _ = f.sync_model(tx)
        num = _numeric_jacobian(lambda e: _perturbed(f, e).sync_model(tx)[0], f.dim, step)
        np.testing.assert_allclose(H, num, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_eqf_transition_matrix(n_clocks):
    _, f = _filters(n_clocks)
    A, _, _ = f.linearization(GYRO, ACCEL, DT)

    def step(eps):
        f_true, f_hat = _perturbed(f, eps), copy.deepcopy(f)
        f_true.propagate(GYRO, ACCEL, DT)
        f_hat.propagate(GYRO, ACCEL, DT)
        return _eqf_error(f_true, f_hat)

    num = _numeric_jacobian(step, f.dim, 1e-6)
    np.testing.assert_allclose(num, A, atol=2e-7)


def test_eqf_input_matrix():
    _, f = _filters(0)
    _, B, _ = f.linearization(GYRO, ACCEL, DT)

    def step(eta):
        f_true, f_hat = copy.deepcopy(f), copy.deepcopy(f)
        f_true.b = f.b + eta  # the truth sees the input minus eta
        f_true.propagate(GYRO, ACCEL, DT)
        f_hat.propagate(GYRO, ACCEL, DT)
        return _eqf_error(f_true, f_hat)[f.core_ext]  # the bias rows carry eta itself

    num = _numeric_jacobian(step, 10, 1e-6)
    np.testing.assert_allclose(num, B[f.core_ext, :10], atol=1e-8)


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_eqf_physical_jacobian_and_error(n_clocks):
    _, f = _filters(n_clocks)
    f.b[6:9] = 0.0  # the physical state has no virtual bias
    x_hat = f.estimate()
    J = f.physical_jacobian()[:-3]
    step = _eqf_steps(f)
    step[9] = step[N_CORE:f.ia] = 1e-9  # position is ~1 m, so smaller time steps drown in round-off
    num = _numeric_jacobian(lambda e: x_hat.error_to(_perturbed(f, e).estimate()), f.dim, step)
    np.testing.assert_allclose(J, num, rtol=1e-5, atol=1e-6)
    eps = 1e-3 * np.random.default_rng(0).normal(size=f.dim)
    eps[16:19] = 0.0  # keep b_nu = 0 so the truth is a physical state
    eps[9], eps[N_CORE:f.ia] = 1e-10, 1e-10
    g = _perturbed(f, eps)
    np.testing.assert_allclose(f.error_to(g.F, g.b, g.A, g.ac, g.ak), eps, atol=1e-12)


def _gauge_eqf(f):
    """Rotation about gravity and translation along x of the whole scene, in eps."""
    yaw, shift = np.zeros(f.dim), np.zeros(f.dim)
    yaw[2] = 1.0
    shift[6] = 1.0
    shift[f.ia::3] = 1.0
    return yaw, shift


def test_eqf_gauge_directions_are_unobservable():
    _, f = _filters(3)
    A, _, _ = f.linearization(GYRO, ACCEL, DT)
    _, H = f.range_model(LEVER)
    for d in _gauge_eqf(f):
        np.testing.assert_allclose(A @ d, d, atol=1e-12)
        np.testing.assert_allclose(H @ d, 0.0, atol=1e-9)
        for tx in range(len(ANCHORS)):
            np.testing.assert_allclose(f.sync_model(tx)[1] @ d, 0.0, atol=1e-12)


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_ekf_range_and_sync_jacobians(n_clocks):
    f, _ = _filters(n_clocks)
    x = f.x
    # The readings include C delta ~ 2e5 m, so the steps must be large enough to beat round-off.
    step = np.full(x.dim, 1e-4)
    step[15] = step[17:17 + 2 * n_clocks] = 1e-13

    def at(e):
        g = copy.deepcopy(f)
        g.x = x.retract(e)
        return g

    num = _numeric_jacobian(lambda e: at(e).range_model(LEVER)[0], x.dim, step)
    np.testing.assert_allclose(f.range_model(LEVER)[1], num, rtol=1e-5, atol=1e-5)
    for tx in range(len(ANCHORS)):
        num = _numeric_jacobian(lambda e: at(e).sync_model(tx)[0], x.dim, step)
        np.testing.assert_allclose(f.sync_model(tx)[1], num, rtol=1e-5, atol=1e-5)


def test_ekf_yaw_direction_depends_on_the_estimate():
    """The EKF's unobservable yaw direction holds at one linearization point, but it
    moves with the estimate, unlike the EqF's."""
    f, _ = _filters(0)
    x = f.x

    def yaw_direction(x):
        d = np.zeros(x.dim)
        ez = np.array([0.0, 0.0, 1.0])
        d[0:3] = x.nav.R.T @ ez
        d[3:6] = np.cross(ez, x.nav.v)
        d[6:9] = np.cross(ez, x.nav.p)
        d[x.nav.dim:] = np.cross(ez, x.anchors).ravel()
        return d

    np.testing.assert_allclose(f.range_model(np.zeros(3))[1] @ yaw_direction(x), 0.0, atol=1e-12)
    moved = copy.deepcopy(f)
    moved.x = x.retract(np.r_[np.zeros(6), 0.2, -0.1, 0.0, np.zeros(x.dim - 9)])
    assert np.abs(f.range_model(np.zeros(3))[1] @ yaw_direction(moved.x)).max() > 1e-2


def test_initial_covariance_maps_back():
    P0 = prior_covariance(SIGMA, 3, len(ANCHORS))
    f = EqfSlam(_state(3), P0, NOISE)
    d = np.sqrt(np.diag(P0))  # compare in units of each state's prior std
    np.testing.assert_allclose(f.physical_covariance() / np.outer(d, d), P0 / np.outer(d, d), atol=1e-8)
    assert frame_from_estimate(f.estimate().nav).shape == (5, 5)
    assert gal3.DIM == 10


def _gauge_ekf(nav, anchors, dim):
    """Columns: rotation of the whole scene about gravity, then its translations
    along x, y, z, in the EKF's coordinates at the point (nav, anchors)."""
    ez, k = np.array([0.0, 0.0, 1.0]), nav.dim
    yaw = np.zeros(dim)
    yaw[0:3], yaw[3:6], yaw[6:9] = nav.R.T @ ez, np.cross(ez, nav.v), np.cross(ez, nav.p)
    yaw[k:] = np.cross(ez, anchors).ravel()
    cols = [yaw]
    for axis in range(3):
        t = np.zeros(dim)
        t[6 + axis] = 1.0
        t[k + axis::3] = 1.0
        cols.append(t)
    return np.array(cols).T


def _moved(x, seed):
    """x after an update-sized correction of the drone and the anchors."""
    rng, dx = np.random.default_rng(seed), np.zeros(x.dim)
    dx[0:9] = rng.normal(scale=0.05, size=9)
    dx[x.nav.dim:] = rng.normal(scale=0.1, size=x.dim - x.nav.dim)
    return x.retract(dx)


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_fej_jacobians_keep_the_gauge_unobservable(n_clocks):
    """After an update moves the estimate, FEJ's Jacobians are still blind to the
    gauge at its linearization point and carry it to the propagated point; the
    plain EKF's are not."""
    x = _state(n_clocks)
    P0 = prior_covariance(SIGMA, n_clocks, len(ANCHORS))
    fej = EkfFej(x, P0, NOISE)
    fej.x = _moved(x, 1)  # an update: the estimate moves, the linearization point does not
    N, k = _gauge_ekf(fej.lin, fej.anchors0, x.dim), x.nav.dim
    np.testing.assert_allclose(fej.range_model(LEVER)[1] @ N, 0.0, atol=1e-9)
    for tx in range(len(ANCHORS)):
        np.testing.assert_allclose(fej.sync_model(tx)[1] @ N, 0.0, atol=1e-12)
    nav, F, _ = fej.transition(GYRO, ACCEL, DT)
    np.testing.assert_allclose(F @ N[:k], _gauge_ekf(nav, fej.anchors0, x.dim)[:k], atol=1e-12)

    ekf = EkfSlam(fej.x, P0, NOISE)  # Jacobians at the moved estimate
    assert np.abs(ekf.range_model(LEVER)[1] @ N).max() > 1e-2
    _, F, _ = ekf.transition(GYRO, ACCEL, DT)
    assert np.abs(F @ N[:k] - _gauge_ekf(nav, fej.anchors0, x.dim)[:k]).max() > 1e-3


@pytest.mark.parametrize("n_clocks", [0, 3])
def test_fej_never_gains_gauge_information(n_clocks):
    """Without process noise, the information N^T P^-1 N about the gauge never
    changes in the FEJ-EKF; the plain EKF's grows."""
    x = _state(n_clocks)
    P0 = prior_covariance(SIGMA, n_clocks, len(ANCHORS))
    s = np.sqrt(np.diag(P0))  # scale every state by its prior std for the solve

    def info(f, nav, anchors):
        N = _gauge_ekf(nav, anchors, x.dim) / s[:, None]
        return N.T @ np.linalg.solve(f.P / np.outer(s, s), N)

    fej, ekf = EkfFej(x, P0, NOISE), EkfSlam(x, P0, NOISE)
    for f in (fej, ekf):
        f.q_input[:], f.q_walk[:], f.q_clock[:] = 0.0, 0.0, 0.0
    info0 = info(fej, x.nav, x.anchors)
    rng = np.random.default_rng(2)
    for step in range(40):
        noise_range, noise_sync = rng.normal(scale=0.1, size=(2, len(ANCHORS)))
        tx = step % len(ANCHORS)
        for f in (fej, ekf):
            for _ in range(4):
                f.propagate(GYRO, ACCEL, DT)
            f.update(f.range_model(LEVER)[0] + noise_range, LEVER)
            if n_clocks:
                h, _, rx = f.sync_model(tx)
                y = np.full(len(ANCHORS), np.nan)
                y[rx] = h + noise_sync[rx]
                f.update_sync(y, tx)
    np.testing.assert_allclose(info(fej, fej.lin, fej.anchors0), info0, rtol=1e-6, atol=1e-6 * np.abs(info0).max())
    assert info(ekf, ekf.x.nav, ekf.x.anchors)[0, 0] > 1.02 * info0[0, 0]  # 0.8 s of flight
