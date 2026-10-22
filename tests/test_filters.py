"""Analytic Jacobians of both filters against finite differences of their exact models."""

import numpy as np
import pytest

from gal3uwb import gal3, lie
from gal3uwb.filters import ekf, eqf
from gal3uwb.filters.common import Estimate, FilterNoise
from gal3uwb.params import ClockNoise, ImuNoise

RNG = np.random.default_rng(1)
GYRO, ACCEL, DT = np.array([0.3, -0.2, 0.5]), np.array([0.4, -0.3, 9.6]), 0.005
ANCHORS = np.array([[3.0, 3.0, 2.8], [-3.0, 3.0, 0.2], [3.0, -3.0, 0.2]])
LEVER = np.array([0.13, -0.17, -0.05])
NOISE = FilterNoise(ImuNoise(), ClockNoise(), 0.1)


def _state():
    return Estimate(lie.exp(np.array([0.1, -0.2, 0.7])), np.array([1.5, -0.8, 0.2]), np.array([1.2, 0.4, 1.1]),
                    np.array([0.01, -0.02, 0.005]), np.array([0.05, -0.03, 0.02]), 7e-4, 2e-5)


def _numeric_jacobian(fun, n, step):
    cols = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = step[i] if np.ndim(step) else step
        cols.append((fun(d) - fun(-d)) / (2 * d[i]))
    return np.array(cols).T


# --- EKF ---------------------------------------------------------------------------


def test_ekf_transition_matrix():
    x = _state()
    x_new, F, _ = ekf.propagate(x, GYRO, ACCEL, DT)
    num = _numeric_jacobian(lambda e: x_new.error_to(ekf.propagate(x.retract(e), GYRO, ACCEL, DT)[0]), 17, 1e-6)
    np.testing.assert_allclose(F, num, atol=1e-7)


def test_ekf_input_matrix():
    x = _state()
    x_new, _, F_input = ekf.propagate(x, GYRO, ACCEL, DT)
    # A constant input error eta acts like a bias error of the same size.
    num = _numeric_jacobian(lambda eta: x_new.error_to(ekf.propagate(x, GYRO - eta[:3], ACCEL - eta[3:], DT)[0]),
                            6, 1e-6)
    np.testing.assert_allclose(F_input, num, atol=1e-7)


def test_ekf_range_jacobian():
    x = _state()
    _, H = ekf.range_model(x, ANCHORS, LEVER)
    # y carries C delta ~ 2e5 m, so spatial steps must be large enough to stay above rounding.
    step = np.r_[np.full(15, 1e-3), 1e-12, 1e-3]
    num = _numeric_jacobian(lambda e: ekf.range_model(x.retract(e), ANCHORS, LEVER)[0], 17, step)
    np.testing.assert_allclose(H, num, rtol=1e-5, atol=1e-5)


# --- EqF ---------------------------------------------------------------------------


def _eqf(F, b):
    f = eqf.Eqf.__new__(eqf.Eqf)
    f.F, f.b, f.noise = F.copy(), b.copy(), NOISE
    f.ac, f.ak = np.zeros(0), np.zeros(0)
    f.P = np.eye(eqf.DIM)
    f.S = np.zeros((eqf.DIM, eqf.DIM))
    return f


def _perturbed(f, eps):
    """The state exp(eps) X_hat, as an Eqf object."""
    X = gal3.g_compose(gal3.g_exp(eps), (f.F, -gal3.Ad(f.F) @ f.b))
    return _eqf(X[0], -gal3.Ad(gal3.inverse(X[0])) @ X[1])


def _error(f_true, f_hat):
    X_true = (f_true.F, -gal3.Ad(f_true.F) @ f_true.b)
    return gal3.g_log(gal3.g_compose(X_true, (gal3.inverse(f_hat.F), f_hat.b)))


def _eqf_state():
    x = _state()
    f = _eqf(eqf.frame_from_estimate(x), np.r_[x.gyro_bias, x.accel_bias, 1e-3, -2e-3, 5e-4, x.rho])
    return f


def _eqf_linearization():
    f = _eqf_state()
    f.P = np.zeros((eqf.DIM, eqf.DIM))
    # Recover A from propagating P = basis outer products is awkward; rebuild it directly.
    F, b = f.F, f.b
    u = np.r_[GYRO, ACCEL, 0.0, 0.0, 0.0, 1.0] - b
    Ad_F = gal3.Ad(F)
    w_ring = Ad_F @ u * DT
    A1 = gal3.Ad(gal3.exp(-eqf.G_N * DT))
    J = gal3.left_jacobian(w_ring)
    A = np.zeros((20, 20))
    A[:10, :10], A[:10, 10:], A[10:, 10:] = A1, A1 @ J * DT, A1 @ gal3.Ad(gal3.exp(w_ring))
    return f, A, -A1 @ J @ Ad_F * DT


def test_eqf_propagation_matches_filter_code():
    """The A and Q built in Eqf.propagate equal the test's reconstruction."""
    f, A, B_input = _eqf_linearization()
    P0 = np.diag(RNG.uniform(0.5, 2.0, 20))
    g = _eqf_state()
    g.P = P0.copy()
    g.propagate(GYRO, ACCEL, DT)
    np.testing.assert_allclose(g.P, A @ P0 @ A.T, atol=1e-12)


def test_eqf_state_matrix():
    f, A, _ = _eqf_linearization()

    def step(eps):
        f_true, f_hat = _perturbed(f, eps), _eqf(f.F, f.b)
        f_true.propagate(GYRO, ACCEL, DT)
        f_hat.propagate(GYRO, ACCEL, DT)
        return _error(f_true, f_hat)

    num = _numeric_jacobian(step, 20, 1e-6)
    np.testing.assert_allclose(num, A, atol=1e-7)


def test_eqf_input_matrix():
    f, _, B_input = _eqf_linearization()

    def step(eta):
        f_true, f_hat = _eqf(f.F, f.b), _eqf(f.F, f.b)
        f_true.b = f.b + eta  # the truth sees the input minus eta
        f_true.propagate(GYRO, ACCEL, DT)
        f_hat.propagate(GYRO, ACCEL, DT)
        return _error(f_true, f_hat)[:10]

    num = _numeric_jacobian(step, 10, 1e-6)
    np.testing.assert_allclose(num, B_input, atol=1e-7)


def test_eqf_range_jacobian():
    f = _eqf_state()
    _, H = f.range_model(ANCHORS, LEVER)
    step = np.r_[np.full(9, 1e-3), 1e-12, np.full(10, 1e-3)]
    num = _numeric_jacobian(lambda eps: _perturbed(f, eps).range_model(ANCHORS, LEVER)[0], 20, step)
    np.testing.assert_allclose(H, num, rtol=1e-5, atol=1e-5)


def test_eqf_physical_jacobian():
    f = _eqf_state()
    x_hat = f.estimate()
    J = eqf.physical_jacobian(f.F, f.b)

    def phys(eps):
        g = _perturbed(f, eps)
        return np.r_[x_hat.error_to(g.estimate()), g.b[6:9] - f.b[6:9]]

    step = np.r_[np.full(9, 1e-6), 1e-9, np.full(10, 1e-6)]
    np.testing.assert_allclose(_numeric_jacobian(phys, 20, step), J, rtol=1e-6, atol=1e-7)


def test_eqf_native_error_inverts_retraction():
    f = _eqf_state()
    eps = 1e-3 * RNG.normal(size=20)
    eps[16:19] = 0.0
    truth_filter = _perturbed(f, eps)
    # native_error assumes b_nu = 0 in truth, so zero it in the perturbed state first.
    truth_filter.b[6:9] = 0.0
    truth = truth_filter.estimate()
    expected = _error(truth_filter, f)[eqf._PHYSICAL]
    np.testing.assert_allclose(f.native_error(truth)[0], expected, atol=1e-12)


@pytest.mark.parametrize("cls", [ekf.Ekf, eqf.Eqf])
def test_update_reduces_error_on_noiseless_measurement(cls):
    truth = _state()
    P0 = np.diag(np.r_[np.full(3, 0.02**2), np.full(3, 0.1**2), np.full(3, 0.3**2), np.full(3, 0.01**2),
                       np.full(3, 0.05**2), (1.0 / 3e8) ** 2, 20e-6**2])
    x0 = truth.retract(np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.1, np.zeros(6), 1e-9, 0.0])
    f = cls(x0, P0, NOISE)
    anchors = np.array([[x, y, z] for z in (0.2, 2.8) for x in (-3, 3) for y in (-3, 3)], dtype=float)
    y = ekf.range_model(truth, anchors, LEVER)[0]
    before = np.linalg.norm(f.estimate().p - truth.p)
    for _ in range(3):
        f.update(y, anchors, LEVER)
    assert np.linalg.norm(f.estimate().p - truth.p) < 0.1 * before


# --- Free-running anchors ---------------------------------------------------------

ANCHORS4 = np.array([[3.0, 3.0, 2.8], [-3.0, 3.0, 0.2], [3.0, -3.0, 0.2], [-3.0, -3.0, 2.8]])


def _state_with_anchors():
    x = _state()
    return Estimate(x.R, x.v, x.p, x.gyro_bias, x.accel_bias, x.delta, x.rho, np.array([3e-4, -2e-4, 5e-4]),
                    np.array([1e-5, -3e-5, 2e-5]))


def test_clock_noise_covariance_matches_device_noise():
    from gal3uwb.filters.common import clock_noise_covariance

    n, q_o, q_s = 3, 2e-19, 5e-19
    # Device noise [tau, o_0..o_n, s_tag, s_0..s_n] -> [delta, rho, Delta_1..n, kappa_1..n].
    M = np.zeros((2 + 2 * n, 2 * (n + 2)))
    M[0, 1], M[0, 0] = 1.0, -1.0  # delta = o_0 - tau
    M[1, n + 2], M[1, n + 3] = 1.0, -1.0  # rho = s_tag - s_0
    for i in range(1, n + 1):
        M[1 + i, 1 + i], M[1 + i, 1] = 1.0, -1.0  # Delta_i = o_i - o_0
        M[1 + n + i, n + 3 + i], M[1 + n + i, n + 3] = 1.0, -1.0  # kappa_i = s_i - s_0
    D = np.diag(np.r_[np.full(n + 2, q_o), np.full(n + 2, q_s)])
    np.testing.assert_allclose(clock_noise_covariance(ClockNoise(q_o, q_s), n), M @ D @ M.T, rtol=1e-12)


def test_ekf_transition_matrix_with_anchor_clocks():
    x = _state_with_anchors()
    x_new, F, _ = ekf.propagate(x, GYRO, ACCEL, DT)
    num = _numeric_jacobian(lambda e: x_new.error_to(ekf.propagate(x.retract(e), GYRO, ACCEL, DT)[0]), x.dim, 1e-6)
    np.testing.assert_allclose(F, num, atol=1e-7)


@pytest.mark.parametrize("cls", [ekf.Ekf, eqf.Eqf])
def test_range_and_sync_jacobians_with_anchor_clocks(cls):
    """Both filters' Jacobians, mapped to physical coordinates, match finite differences
    of the physical measurement model."""
    x = _state_with_anchors()
    f = cls(x, np.eye(x.dim) * 1e-4, NOISE)
    J = np.eye(x.dim) if cls is ekf.Ekf else np.linalg.inv(eqf.physical_jacobian(f.F, f.b, 3))[:, :x.dim]
    step = np.r_[np.full(15, 1e-3), 1e-12, 1e-3, np.full(3, 1e-12), np.full(3, 1e-3)]
    _, H = f.range_model(ANCHORS4, LEVER) if cls is eqf.Eqf else ekf.range_model(x, ANCHORS4, LEVER)
    num = _numeric_jacobian(lambda e: ekf.range_model(x.retract(e), ANCHORS4, LEVER)[0], x.dim, step)
    np.testing.assert_allclose(H @ J, num, rtol=1e-5, atol=1e-5)
    for tx in (0, 2):
        offsets = ekf.anchor_offsets(x, 4)
        cols = ekf.anchor_columns(x, 4) if cls is ekf.Ekf else f._anchor_columns(4)
        _, H, _ = ekf.sync_model(len(f.P), offsets, cols, tx, ANCHORS4)
        num = _numeric_jacobian(
            lambda e: ekf.sync_model(x.dim, ekf.anchor_offsets(x.retract(e), 4), ekf.anchor_columns(x, 4), tx,
                                     ANCHORS4)[0], x.dim, step)
        np.testing.assert_allclose(H @ J, num, rtol=1e-5, atol=1e-5)
