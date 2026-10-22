"""Gal(3) and its tangent group Gal(3) x| gal(3), as plain numpy arrays.

Gal(3) elements are 5x5 matrices [[R, v, p], [0, 1, t], [0, 0, 1]]. Tangent
vectors follow GTSAM's order [omega(0:3), nu(3:6), rho(6:9), s(9)], with hat

    [[omega^, nu, rho], [0, 0, s], [0, 0, 0]].

The tangent group G = Gal(3) x| gal(3) (Delama et al.) has elements (A, a) with
A a 5x5 matrix and a a 10-vector, product (A, a)(B, b) = (AB, a + Ad_A b), and
20-dim tangent vectors (x, y) with exp(x, y) = (exp x, J(x) y).
"""

import numpy as np

from gal3uwb import lie

DIM = 10
ROT, NU, RHO, TIME = slice(0, 3), slice(3, 6), slice(6, 9), 9


def hat(xi):
    X = np.zeros((5, 5))
    X[:3, :3] = lie.hat(xi[ROT])
    X[:3, 3] = xi[NU]
    X[:3, 4] = xi[RHO]
    X[3, 4] = xi[TIME]
    return X


def vee(X):
    return np.concatenate([lie.vee(X[:3, :3]), X[:3, 3], X[:3, 4], [X[3, 4]]])


def make(R, v, p, t):
    X = np.eye(5)
    X[:3, :3], X[:3, 3], X[:3, 4], X[3, 4] = R, v, p, t
    return X


def parts(X):
    """(R, v, p, t) views of a Gal(3) matrix."""
    return X[:3, :3], X[:3, 3], X[:3, 4], X[3, 4]


def inverse(X):
    R, v, p, t = parts(X)
    Rt = R.T
    return make(Rt, -Rt @ v, -Rt @ (p - t * v), -t)


def exp(xi):
    phi, nu, rho, s = xi[ROT], xi[NU], xi[RHO], xi[TIME]
    G1 = lie.gamma1(phi)
    return make(lie.exp(phi), G1 @ nu, G1 @ rho + s * (lie.gamma2(phi) @ nu), s)


def log(X):
    R, v, p, t = parts(X)
    phi = lie.log(R)
    G1_inv = np.linalg.inv(lie.gamma1(phi))
    nu = G1_inv @ v
    return np.concatenate([phi, nu, G1_inv @ (p - t * (lie.gamma2(phi) @ nu)), [t]])


def Ad(X):
    """Adjoint matrix: hat(Ad(X) xi) = X hat(xi) X^-1."""
    R, v, p, t = parts(X)
    M = np.zeros((DIM, DIM))
    M[ROT, ROT] = R
    M[NU, ROT] = lie.hat(v) @ R
    M[NU, NU] = R
    M[RHO, ROT] = lie.hat(p - t * v) @ R
    M[RHO, NU] = -t * R
    M[RHO, RHO] = R
    M[RHO, TIME] = v
    M[TIME, TIME] = 1.0
    return M


def ad(xi):
    """Lie bracket matrix: ad(a) b = vee([hat(a), hat(b)])."""
    M = np.zeros((DIM, DIM))
    W = lie.hat(xi[ROT])
    M[ROT, ROT] = W
    M[NU, ROT] = lie.hat(xi[NU])
    M[NU, NU] = W
    M[RHO, ROT] = lie.hat(xi[RHO])
    M[RHO, NU] = -xi[TIME] * np.eye(3)
    M[RHO, RHO] = W
    M[RHO, TIME] = xi[NU]
    return M


def series_jacobian(adx, tol=1e-15, max_terms=40):
    """sum_k adx^k / (k+1)!, the left Jacobian for an ad matrix of moderate norm."""
    out = np.eye(len(adx))
    term = np.eye(len(adx))
    for k in range(1, max_terms):
        term = term @ adx / (k + 1)
        out += term
        if np.abs(term).max() < tol:
            return out
    raise ValueError(f"left Jacobian series did not converge (|ad| = {np.abs(adx).max():.3g})")


def left_jacobian(xi):
    """J such that exp(xi + d) ~= exp(J d) exp(xi)."""
    return series_jacobian(ad(xi))


# --- Tangent group G = Gal(3) x| gal(3) ------------------------------------------------


def g_compose(X, Y):
    return X[0] @ Y[0], X[1] + Ad(X[0]) @ Y[1]


def g_inverse(X):
    A_inv = inverse(X[0])
    return A_inv, -Ad(A_inv) @ X[1]


def g_exp(z):
    x, y = z[:DIM], z[DIM:]
    return exp(x), left_jacobian(x) @ y


def g_log(X):
    x = log(X[0])
    return np.concatenate([x, np.linalg.solve(left_jacobian(x), X[1])])


def g_ad(z):
    x, y = z[:DIM], z[DIM:]
    M = np.zeros((2 * DIM, 2 * DIM))
    M[:DIM, :DIM] = M[DIM:, DIM:] = ad(x)
    M[DIM:, :DIM] = ad(y)
    return M


def g_left_jacobian(z):
    return series_jacobian(g_ad(z))

