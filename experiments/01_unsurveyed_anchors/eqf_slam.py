"""EqF with the anchor positions inside the symmetry group.

Experiment 00's EqF has the state (F, b) on Gal(3) x| gal(3). Here the anchor
positions join F as extra columns, the way landmarks join SE_2(3) in invariant
EKF-SLAM:

    X = [[R, v_F, p_F, a_0 ... a_{N-1}],
         [0,   1, delta,  0 ...     0],
         [0,   0,     1,  0 ...     0],
         [0,   0,     0,  I_N         ]]

The product rotates and translates the anchor columns with the drone:
X1 X2 has anchor columns R1 a2_i + a1_i. The symmetry group is
(Gal(3) x| R^{3N}) x| gal(3); the biases see only the Gal(3) part. Anchor
clocks (step 3) stay additive, as in experiment 00.

The error is X = exp(eps) X_hat with
    eps = [Gal(3) part (10), bias part (10), anchor clocks (2n), anchor columns alpha (3N)].
Rotating the whole scene about gravity, or translating it, is a left
multiplication, so in eps it is the same vector whatever the estimate:
(theta = e_z, 0, ...) and (rho = t, alpha_i = t). Both propagation and every
measurement leave these directions exactly unobservable, which is the property
the EKF lacks.
"""

import numpy as np

from gal3uwb import gal3, lie
from gal3uwb.filters.eqf import G_N, Eqf, estimate_from_state, frame_from_estimate, physical_jacobian
from gal3uwb.params import C, GRAVITY

from model import SlamEstimate

N_GAL = gal3.DIM  # 10
N_CORE = 2 * N_GAL  # Gal(3) x| gal(3)
_PHYSICAL = np.r_[0:16, 19]  # eps of Gal(3) x| gal(3) without the virtual bias b_nu


def anchor_ad(head_ad, theta, alpha):
    """ad matrix of a group whose head part has ad matrix head_ad (its first three
    coordinates are the rotation theta) followed by anchor columns alpha (N, 3).
    The bracket of the anchor columns is [X, Y]_i = theta_X x alpha_Y,i - theta_Y x alpha_X,i."""
    h, N = len(head_ad), len(alpha)
    M = np.zeros((h + 3 * N, h + 3 * N))
    M[:h, :h] = head_ad
    W = lie.hat(theta)
    for i in range(N):
        s = slice(h + 3 * i, h + 3 * i + 3)
        M[s, 0:3] = lie.hat(alpha[i])
        M[s, s] = W
    return M


def embed(A):
    """Ad_X of a Gal(3)-only tangent vector xi has anchor columns a_i x (Ad_F xi)_theta.
    This (10 + 3N) x 10 matrix maps Ad_F xi to [Ad_F xi, those anchor columns]."""
    N = len(A)
    E = np.zeros((N_GAL + 3 * N, N_GAL))
    E[:N_GAL] = np.eye(N_GAL)
    for i in range(N):
        E[N_GAL + 3 * i:N_GAL + 3 * i + 3, 0:3] = lie.hat(A[i])
    return E


class EqfSlam:
    name = "EqF"

    def __init__(self, x0: SlamEstimate, P0, noise):
        """P0 is the covariance in the physical coordinates of model.py; it is mapped to eps."""
        self.F = frame_from_estimate(x0.nav)
        self.b = np.r_[x0.nav.gyro_bias, x0.nav.accel_bias, 0.0, 0.0, 0.0, x0.nav.rho]
        self.ac, self.ak = x0.nav.anchor_offset.copy(), x0.nav.anchor_skew.copy()
        self.A = x0.anchors.copy()
        self.noise = noise
        n, N = len(self.ac), len(self.A)
        self.n, self.N = n, N
        self.ia = N_CORE + 2 * n  # first anchor column in eps
        self.dim = self.ia + 3 * N
        self.core_ext = np.r_[0:N_GAL, self.ia:self.dim]  # Gal(3) part and anchor columns
        self.group = np.r_[0:N_CORE, self.ia:self.dim]  # everything but the additive anchor clocks
        P_phys = np.zeros((self.dim, self.dim))
        P_phys[:-3, :-3] = P0
        P_phys[-3:, -3:] = noise.nu_sigma0**2 * np.eye(3)
        J_inv = np.linalg.inv(self.physical_jacobian())
        self.P = J_inv @ P_phys @ J_inv.T
        self.S = Eqf.noise_psd(noise, n)

    # --- group operations ------------------------------------------------------------

    def apply(self, delta):
        """X_hat <- exp(delta) X_hat; the anchor clocks are additive."""
        X_hat = (self.F, -gal3.Ad(self.F) @ self.b)
        self.F, X_b = gal3.g_compose(gal3.g_exp(delta[:N_CORE]), X_hat)
        self.b = -gal3.Ad(gal3.inverse(self.F)) @ X_b
        theta, alpha = delta[0:3], delta[self.ia:].reshape(self.N, 3)
        # exp(delta) has rotation Exp(theta) and anchor columns J_SO3(theta) alpha_i.
        self.A = self.A @ lie.exp(theta).T + alpha @ lie.gamma1(theta).T
        self.ac = self.ac + delta[N_CORE:N_CORE + self.n]
        self.ak = self.ak + delta[N_CORE + self.n:self.ia]

    def error_to(self, F_true, b_true, A_true, ac_true, ak_true):
        """eps with X_true = exp(eps) X_hat (anchor clocks: truth minus estimate)."""
        X_true = (F_true, -gal3.Ad(F_true) @ b_true)
        core = gal3.g_log(gal3.g_compose(X_true, (gal3.inverse(self.F), self.b)))
        R_e = F_true[:3, :3] @ self.F[:3, :3].T
        alpha = np.linalg.solve(lie.gamma1(core[0:3]), (A_true - self.A @ R_e.T).T).T
        return np.r_[core, ac_true - self.ac, ak_true - self.ak, alpha.ravel()]

    # --- filter ----------------------------------------------------------------------

    def linearization(self, gyro, accel, dT):
        """Error transition A and input map B over one IMU step (columns of B:
        [input noise (10), bias random walk (10), anchor clock noise (2n)])."""
        n, N, dim = self.n, self.N, self.dim
        u = np.r_[gyro, accel, 0.0, 0.0, 0.0, 1.0] - self.b
        Ad_F = gal3.Ad(self.F)
        w_ring = Ad_F @ u * dT
        E = embed(self.A)
        w_ext = E @ w_ring
        J = gal3.series_jacobian(anchor_ad(gal3.ad(w_ring), w_ring[0:3], w_ext[N_GAL:].reshape(N, 3)))
        A1 = gal3.Ad(gal3.exp(-G_N * dT))  # the gravity factor has no anchor columns and no rotation
        A1_ext = np.eye(N_GAL + 3 * N)
        A1_ext[:N_GAL, :N_GAL] = A1
        A2 = A1 @ gal3.Ad(gal3.exp(w_ring))
        M = A1_ext @ J @ E
        ce, bias = self.core_ext, slice(N_GAL, N_CORE)
        A = np.eye(dim)
        A[np.ix_(ce, ce)] = A1_ext
        A[ce, bias] = M * dT
        A[bias, bias] = A2
        A[N_CORE:N_CORE + n, N_CORE + n:self.ia] = dT * np.eye(n)
        B = np.zeros((dim, N_CORE + 2 * n))
        B[ce, 0:N_GAL] = -M @ Ad_F * dT
        B[bias, N_GAL:N_CORE] = -A2 @ Ad_F * dT
        B[N_CORE:self.ia, N_CORE:] = dT * np.eye(2 * n)
        return A, B, u

    def propagate(self, gyro, accel, dT):
        A, B, u = self.linearization(gyro, accel, dT)
        self.P = A @ self.P @ A.T + B @ self.S @ B.T / dT
        self.F = gal3.exp(-G_N * dT) @ self.F @ gal3.exp(u * dT)
        self.ac = self.ac + self.ak * dT

    def offsets(self):
        return np.r_[0.0, self.ac] if self.n else np.zeros(self.N)

    def range_model(self, lever_arm):
        R, v_F, p_F, d = gal3.parts(self.F)
        base = p_F + R @ lever_arm
        diff = base + 0.5 * GRAVITY * d * d - self.A
        r = np.linalg.norm(diff, axis=1)
        u = diff / r[:, None]
        H = np.zeros((self.N, self.dim))
        for i in range(self.N):
            H[i, 0:3] = -u[i] @ lie.hat(base - self.A[i])
            H[i, self.ia + 3 * i:self.ia + 3 * i + 3] = -u[i]
            if self.n and i > 0:
                H[i, N_CORE + i - 1] = C
        H[:, 3:6] = d * u
        H[:, 6:9] = u
        H[:, 9] = C + d * (u @ GRAVITY)
        return C * (d + self.offsets()) + r, H

    def sync_model(self, tx):
        rx = np.array([j for j in range(self.N) if j != tx])
        diff = self.A[rx] - self.A[tx]
        dist = np.linalg.norm(diff, axis=1)
        u = diff / dist[:, None]
        off = self.offsets()
        H = np.zeros((len(rx), self.dim))
        for row, j in enumerate(rx):
            if self.n and j > 0:
                H[row, N_CORE + j - 1] += C
            if self.n and tx > 0:
                H[row, N_CORE + tx - 1] -= C
            H[row, 0:3] = -u[row] @ lie.hat(self.A[j] - self.A[tx])  # zero: u is along a_j - a_tx
            H[row, self.ia + 3 * j:self.ia + 3 * j + 3] += u[row]
            H[row, self.ia + 3 * tx:self.ia + 3 * tx + 3] -= u[row]
        return dist + C * (off[rx] - off[tx]), H, rx

    def update(self, y, lever_arm):
        h, H = self.range_model(lever_arm)
        self._correct(y - h, H)

    def update_sync(self, y, tx):
        h, H, rx = self.sync_model(tx)
        self._correct(y[rx] - h, H)

    def _correct(self, r, H):
        R = self.noise.sigma_range**2 * np.eye(len(r))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        delta = K @ r
        I_KH = np.eye(self.dim) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        # Reset (Ge et al.): re-express the covariance around exp(delta) X_hat.
        g = self.group
        dg = delta[g]
        J = np.eye(self.dim)
        J[np.ix_(g, g)] = gal3.series_jacobian(
            anchor_ad(gal3.g_ad(dg[:N_CORE]), dg[0:3], dg[N_CORE:].reshape(self.N, 3)))
        self.P = J @ (0.5 * (P + P.T)) @ J.T
        self.apply(delta)

    # --- evaluation ------------------------------------------------------------------

    def physical_jacobian(self):
        """d(physical error)/d(eps), square: rows are model.py's physical coordinates
        followed by the virtual bias b_nu."""
        n, N, dim = self.n, self.N, self.dim
        J20 = physical_jacobian(self.F, self.b, 0)
        J = np.zeros((dim, dim))
        J[:17, :N_CORE] = J20[:17]
        J[17:17 + 2 * n, N_CORE:self.ia] = np.eye(2 * n)
        ra = 17 + 2 * n
        for i in range(N):
            # a = Exp(theta) a_hat + alpha_i to first order: d a = -a_hat x theta + alpha_i
            J[ra + 3 * i:ra + 3 * i + 3, 0:3] = -lie.hat(self.A[i])
            J[ra + 3 * i:ra + 3 * i + 3, self.ia + 3 * i:self.ia + 3 * i + 3] = np.eye(3)
        J[ra + 3 * N:, :N_CORE] = J20[17:]
        return J

    def estimate(self):
        return SlamEstimate(estimate_from_state(self.F, self.b, self.ac, self.ak), self.A.copy())

    def physical_covariance(self):
        J = self.physical_jacobian()[:-3]
        return J @ self.P @ J.T

    def native_error(self, truth: SlamEstimate):
        """eps for the true state and its covariance, without the virtual bias b_nu."""
        nav = truth.nav
        F_true = frame_from_estimate(nav)
        b_true = np.r_[nav.gyro_bias, nav.accel_bias, 0.0, 0.0, 0.0, nav.rho]
        eps = self.error_to(F_true, b_true, truth.anchors, nav.anchor_offset, nav.anchor_skew)
        idx = np.r_[_PHYSICAL, N_CORE:self.dim]
        return eps[idx], self.P[np.ix_(idx, idx)]
