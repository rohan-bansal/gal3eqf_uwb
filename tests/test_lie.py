import gtsam
import numpy as np
import pytest

from gal3uwb import lie

PHIS = [np.array([1e-9, -2e-9, 3e-9]), np.array([3e-3, -1e-3, 2e-3]), np.array([0.3, -0.7, 0.2]), np.array([2.0, 1.0, -0.5])]


def _integrate(phi, weight, n=4001):
    s = np.linspace(0.0, 1.0, n)
    vals = np.array([weight(si) * gtsam.Rot3.Expmap(si * phi).matrix() for si in s])
    return np.trapezoid(vals, s, axis=0)


@pytest.mark.parametrize("phi", PHIS)
def test_exp_matches_gtsam(phi):
    np.testing.assert_allclose(lie.exp(phi), gtsam.Rot3.Expmap(phi).matrix(), atol=1e-14)


@pytest.mark.parametrize("phi", PHIS + [np.array([0.0, 0.0, np.pi - 1e-8])])
def test_log_inverts_exp(phi):
    np.testing.assert_allclose(lie.log(lie.exp(phi)), phi, atol=1e-7)


@pytest.mark.parametrize("phi", PHIS)
def test_gamma1_is_integral_of_exp(phi):
    np.testing.assert_allclose(lie.gamma1(phi), _integrate(phi, lambda s: 1.0), atol=1e-7)


@pytest.mark.parametrize("phi", PHIS)
def test_gamma2_is_weighted_integral_of_exp(phi):
    np.testing.assert_allclose(lie.gamma2(phi), _integrate(phi, lambda s: 1.0 - s), atol=1e-7)
