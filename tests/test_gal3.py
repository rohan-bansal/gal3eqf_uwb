import gtsam
import numpy as np
import pytest
from scipy.linalg import expm

from gal3uwb import gal3

RNG = np.random.default_rng(0)
XIS = [np.r_[1e-9 * RNG.normal(size=3), RNG.normal(size=7)], 0.3 * RNG.normal(size=10), RNG.normal(size=10)]


@pytest.mark.parametrize("xi", XIS)
def test_exp_matches_matrix_exponential(xi):
    np.testing.assert_allclose(gal3.exp(xi), expm(gal3.hat(xi)), atol=1e-12)


@pytest.mark.parametrize("xi", XIS)
def test_exp_matches_gtsam(xi):
    np.testing.assert_allclose(gal3.exp(xi), gtsam.Gal3.Expmap(xi).matrix(), atol=1e-12)


@pytest.mark.parametrize("xi", XIS)
def test_log_inverts_exp(xi):
    np.testing.assert_allclose(gal3.log(gal3.exp(xi)), xi, atol=1e-10)


@pytest.mark.parametrize("xi", XIS)
def test_inverse(xi):
    X = gal3.exp(xi)
    np.testing.assert_allclose(gal3.inverse(X) @ X, np.eye(5), atol=1e-12)


@pytest.mark.parametrize("xi", XIS)
def test_adjoint_matches_conjugation_and_gtsam(xi):
    X, d = gal3.exp(xi), RNG.normal(size=10)
    np.testing.assert_allclose(gal3.hat(gal3.Ad(X) @ d), X @ gal3.hat(d) @ gal3.inverse(X), atol=1e-12)
    np.testing.assert_allclose(gal3.Ad(X), gtsam.Gal3.Expmap(xi).AdjointMap(), atol=1e-12)


@pytest.mark.parametrize("xi", XIS)
def test_ad_is_bracket(xi):
    d = RNG.normal(size=10)
    A, B = gal3.hat(xi), gal3.hat(d)
    np.testing.assert_allclose(gal3.hat(gal3.ad(xi) @ d), A @ B - B @ A, atol=1e-12)


@pytest.mark.parametrize("xi", XIS[:2])
def test_left_jacobian(xi):
    d = 1e-6 * RNG.normal(size=10)
    lhs = gal3.log(gal3.exp(xi + d) @ gal3.inverse(gal3.exp(xi)))
    np.testing.assert_allclose(lhs, gal3.left_jacobian(xi) @ d, atol=1e-11)


def _g_random(scale):
    return gal3.g_exp(scale * RNG.normal(size=20))


def test_tangent_group_exp_is_one_parameter_subgroup():
    z = 0.4 * RNG.normal(size=20)
    half = gal3.g_exp(0.5 * z)
    full = gal3.g_compose(half, half)
    expected = gal3.g_exp(z)
    np.testing.assert_allclose(full[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(full[1], expected[1], atol=1e-12)


def test_tangent_group_log_and_inverse():
    X = _g_random(0.5)
    np.testing.assert_allclose(gal3.g_exp(gal3.g_log(X))[1], X[1], atol=1e-12)
    eye = gal3.g_compose(X, gal3.g_inverse(X))
    np.testing.assert_allclose(eye[0], np.eye(5), atol=1e-12)
    np.testing.assert_allclose(eye[1], 0.0, atol=1e-12)


def test_tangent_group_left_jacobian():
    z, d = 0.3 * RNG.normal(size=20), 1e-6 * RNG.normal(size=20)
    lhs = gal3.g_log(gal3.g_compose(gal3.g_exp(z + d), gal3.g_inverse(gal3.g_exp(z))))
    np.testing.assert_allclose(lhs, gal3.g_left_jacobian(z) @ d, atol=1e-11)
