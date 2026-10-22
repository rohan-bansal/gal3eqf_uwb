"""SO(3) helpers for scalar (per-step) use in the filters.

Gamma functions follow Barrau & Bonnabel: for constant body rate w and specific
force f over h, with phi = w h,
    R(h) = R Exp(phi)
    v(h) = v + R Gamma1(phi) f h + g h
    p(h) = p + v h + R Gamma2(phi) f h^2 + g h^2 / 2
"""

import math

import numpy as np

_SMALL = 1e-2  # below this angle, use Taylor series


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def _coeffs(theta):
    """Return sin(t)/t, (1-cos t)/t^2, (t-sin t)/t^3, (t^2+2cos t-2)/(2t^4)."""
    t2 = theta * theta
    if theta < _SMALL:
        t4 = t2 * t2
        return (
            1.0 - t2 / 6.0 + t4 / 120.0,
            0.5 - t2 / 24.0 + t4 / 720.0,
            1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0,
            1.0 / 24.0 - t2 / 720.0 + t4 / 40320.0,
        )
    s, c = math.sin(theta), math.cos(theta)
    return s / theta, (1.0 - c) / t2, (theta - s) / (t2 * theta), (t2 + 2.0 * c - 2.0) / (2.0 * t2 * t2)


def exp(phi):
    W = hat(phi)
    a, b, _, _ = _coeffs(math.sqrt(phi @ phi))
    return np.eye(3) + a * W + b * (W @ W)


def log(R):
    cos_theta = min(1.0, max(-1.0, 0.5 * (np.trace(R) - 1.0)))
    theta = math.acos(cos_theta)
    w = vee(R - R.T)
    if theta < _SMALL:
        return 0.5 * w * (1.0 + theta * theta / 6.0)
    if math.pi - theta < 1e-6:
        # Near pi: recover the axis from the symmetric part.
        B = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / math.sqrt(B[k, k])
        return theta * axis * (1.0 if axis @ w >= 0 else -1.0)
    return 0.5 * theta / math.sin(theta) * w


def gamma1(phi):
    """Left Jacobian of SO(3): integral_0^1 Exp(s phi) ds."""
    W = hat(phi)
    _, b, c, _ = _coeffs(math.sqrt(phi @ phi))
    return np.eye(3) + b * W + c * (W @ W)


def gamma2(phi):
    """integral_0^1 (1 - s) Exp(s phi) ds."""
    W = hat(phi)
    _, _, c, d = _coeffs(math.sqrt(phi @ phi))
    return 0.5 * np.eye(3) + c * W + d * (W @ W)
