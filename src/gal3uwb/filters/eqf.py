"""Equivariant filter on Gal(3) x| gal(3), following Delama et al. (RSS 2026).

State xi = (F, b). F = (R, v_F, p_F, delta) in Gal(3) is the Galilean frame of
the drone at tag time T relative to the Earth frame at network time T, so its
time component is the clock offset delta = t - T. Because the Earth frame
accelerates at -g relative to free fall, F's velocity and position differ from
the physical ones by
    v = v_F + g delta,   p = p_F + g delta^2 / 2.
b = (b_omega, b_a, b_nu, b_rho) in gal(3): IMU biases, a virtual velocity-input
bias (zero in truth) and the time-rate bias b_rho, which is the tag skew.

Dynamics in tag time (Delama eq. 24, no Earth rotation):
    dF/dT = -g_N^ F + F (w_N - b)^,  g_N = (0, -g, 0, 1),  w_N = (omega, a, 0, 1),
so d(delta)/dT = -b_rho. The error is e = X X_hat^-1 on the symmetry group with
origin at the identity; the truth is exp(eps) X_hat.

With free-running anchors, anchor 0 is the network reference (delta = A_0 - T)
and the other anchors' clocks Delta_i = A_i - A_0 and skews kappa_i are extra
states on the translation group R^2n, a direct product with Gal(3) x| gal(3).
Their error is additive, as in the EKF.
"""

import numpy as np

from gal3uwb import gal3, lie
from gal3uwb.filters.common import BA, BG, DELTA, POS, RHO, ROT, VEL, Estimate, FilterNoise, clock_noise_covariance
from gal3uwb.filters.ekf import anchor_offsets, sync_model
from gal3uwb.params import C, GRAVITY

N = gal3.DIM  # 10
DIM = 2 * N  # 20: Gal(3) x| gal(3); anchor clocks follow
G_N = np.r_[0.0, 0.0, 0.0, -GRAVITY, 0.0, 0.0, 0.0, 1.0]
_PHYSICAL = np.r_[0:16, 19]  # eps of Gal(3) x| gal(3) without the b_nu coordinates


def frame_from_estimate(x: Estimate):
    d = x.delta
    return gal3.make(x.R, x.v - GRAVITY * d, x.p - 0.5 * GRAVITY * d * d, d)


def estimate_from_state(F, b, anchor_offset=None, anchor_skew=None):
    R, v_F, p_F, d = gal3.parts(F)
    extra = {} if anchor_offset is None else {"anchor_offset": anchor_offset.copy(), "anchor_skew": anchor_skew.copy()}
    return Estimate(R.copy(), v_F + GRAVITY * d, p_F + 0.5 * GRAVITY * d * d, b[0:3].copy(), b[3:6].copy(), d,
                    b[9], **extra)


def physical_jacobian(F, b, n=0):
    """d(physical error)/d(eps) at the estimate, square of size 20 + 2n.

    Rows are the 17 + 2n physical error coordinates (common.py) followed by the
    virtual bias b_nu; columns are eps = (Gal(3) error, bias error, anchor clocks).
    """
    R, v_F, p_F, d = gal3.parts(F)
    J20 = np.zeros((DIM, DIM))
    J20[ROT, 0:3] = R.T
    J20[VEL, 0:3] = -lie.hat(v_F)
    J20[VEL, 3:6] = np.eye(3)
    J20[VEL, 9] = GRAVITY
    J20[POS, 0:3] = -lie.hat(p_F)
    J20[POS, 3:6] = d * np.eye(3)
    J20[POS, 6:9] = np.eye(3)
    J20[POS, 9] = GRAVITY * d
    J20[DELTA, 9] = 1.0
    # Bias error: eps_b = -Ad(F) (b - b_hat) to first order.
    dbias = -gal3.Ad(gal3.inverse(F))
    J20[BG, N:] = dbias[0:3]
    J20[BA, N:] = dbias[3:6]
    J20[RHO, N:] = dbias[9]
    J20[17:20, N:] = dbias[6:9]
    if n == 0:
        return J20
    J = np.zeros((DIM + 2 * n, DIM + 2 * n))
    J[:17, :DIM] = J20[:17]
    J[17:17 + 2 * n, DIM:] = np.eye(2 * n)
    J[17 + 2 * n:, :DIM] = J20[17:]
    return J


class Eqf:
    name = "EqF"

    def __init__(self, x0: Estimate, P0, noise: FilterNoise):
        """P0 is the covariance in physical coordinates (17 + 2n); it is mapped to eps."""
        self.F = frame_from_estimate(x0)
        self.b = np.r_[x0.gyro_bias, x0.accel_bias, 0.0, 0.0, 0.0, x0.rho]
        self.ac, self.ak = x0.anchor_offset.copy(), x0.anchor_skew.copy()
        self.noise = noise
        n = len(self.ac)
        P_phys = np.zeros((DIM + 2 * n, DIM + 2 * n))
        P_phys[:17 + 2 * n, :17 + 2 * n] = P0
        P_phys[17 + 2 * n:, 17 + 2 * n:] = noise.nu_sigma0**2 * np.eye(3)
        J_inv = np.linalg.inv(physical_jacobian(self.F, self.b, n))
        self.P = J_inv @ P_phys @ J_inv.T
        self.S = self.noise_psd(noise, n)

    @staticmethod
    def noise_psd(noise: FilterNoise, n):
        """PSD of [input noise on w_N (10), bias random walk (10), w_Delta (n), w_kappa (n)].

        The time slot of w_N carries the clock offset noise (d(delta)/dT = -b_rho - eta_9)
        and b_rho's random walk is the skew noise; with free-running anchors both
        are correlated with the anchor clocks' noise through anchor 0.
        """
        imu = noise.imu
        clock = clock_noise_covariance(noise.clock, n)  # [delta, rho, Delta, kappa]
        S = np.zeros((DIM + 2 * n, DIM + 2 * n))
        S[np.diag_indices(DIM)] = np.r_[
            np.full(3, imu.gyro_nd**2), np.full(3, imu.accel_nd**2), np.full(3, noise.nu_nd**2), clock[0, 0],
            np.full(3, imu.gyro_bias_rw**2), np.full(3, imu.accel_bias_rw**2), np.full(3, noise.nu_rw**2), clock[1, 1],
        ]
        off, skew = slice(DIM, DIM + n), slice(DIM + n, DIM + 2 * n)
        S[off, off] = clock[2:2 + n, 2:2 + n]
        S[skew, skew] = clock[2 + n:, 2 + n:]
        S[9, off] = S[off, 9] = -clock[0, 2:2 + n]  # eta_9 = -w_delta
        S[19, skew] = S[skew, 19] = clock[1, 2 + n:]
        return S

    @property
    def n_anchor_clocks(self):
        return len(self.ac)

    def propagate(self, gyro, accel, dT):
        F, b, n = self.F, self.b, self.n_anchor_clocks
        u = np.r_[gyro, accel, 0.0, 0.0, 0.0, 1.0] - b
        Ad_F = gal3.Ad(F)
        w_ring = Ad_F @ u * dT
        E_g = gal3.exp(-G_N * dT)
        A1 = gal3.Ad(E_g)
        J = gal3.left_jacobian(w_ring)
        A2 = A1 @ gal3.Ad(gal3.exp(w_ring))
        dim = DIM + 2 * n
        A = np.eye(dim)
        A[:N, :N] = A1
        A[:N, N:DIM] = A1 @ J * dT
        A[N:DIM, N:DIM] = A2
        A[DIM:DIM + n, DIM + n:] = dT * np.eye(n)
        B = np.zeros((dim, dim))
        B[:N, :N] = -A1 @ J @ Ad_F * dT
        B[N:DIM, N:DIM] = -A2 @ Ad_F * dT
        B[DIM:, DIM:] = dT * np.eye(2 * n)
        self.P = A @ self.P @ A.T + B @ self.S @ B.T / dT
        self.F = E_g @ F @ gal3.exp(u * dT)
        self.ac = self.ac + self.ak * dT

    def _anchor_columns(self, n_anchors):
        if not self.n_anchor_clocks:
            return [None] * n_anchors
        return [None] + [DIM + i for i in range(self.n_anchor_clocks)]

    def range_model(self, anchors, lever_arm):
        R, v_F, p_F, d = gal3.parts(self.F)
        p_tag = p_F + 0.5 * GRAVITY * d * d + R @ lever_arm
        diff = p_tag[None, :] - anchors
        r = np.linalg.norm(diff, axis=1)
        u = diff / r[:, None]
        H = np.zeros((len(anchors), DIM + 2 * self.n_anchor_clocks))
        H[:, 0:3] = -u @ lie.hat(p_F + R @ lever_arm)
        H[:, 3:6] = d * u
        H[:, 6:9] = u
        H[:, 9] = C + d * (u @ GRAVITY)
        for i, col in enumerate(self._anchor_columns(len(anchors))):
            if col is not None:
                H[i, col] = C
        return C * (d + anchor_offsets(self.estimate(), len(anchors))) + r, H

    def update(self, y, anchors, lever_arm):
        h, H = self.range_model(anchors, lever_arm)
        self._correct(y - h, H)

    def update_sync(self, y, tx, anchors):
        """Anchor-to-anchor packet from anchor tx; y holds every anchor's timestamp (NaN at tx)."""
        n = len(anchors)
        h, H, rx = sync_model(DIM + 2 * self.n_anchor_clocks, anchor_offsets(self.estimate(), n),
                              self._anchor_columns(n), tx, anchors)
        self._correct(y[rx] - h, H)

    def _correct(self, r, H):
        dim, n = len(self.P), self.n_anchor_clocks
        R = self.noise.sigma_range**2 * np.eye(len(r))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        delta = K @ r
        I_KH = np.eye(dim) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        J = np.eye(dim)
        J[:DIM, :DIM] = gal3.g_left_jacobian(delta[:DIM])  # reset (Ge et al.); anchor clocks are additive
        self.P = J @ (0.5 * (P + P.T)) @ J.T
        X_hat = (self.F, -gal3.Ad(self.F) @ self.b)
        self.F, X_b = gal3.g_compose(gal3.g_exp(delta[:DIM]), X_hat)
        self.b = -gal3.Ad(gal3.inverse(self.F)) @ X_b
        self.ac = self.ac + delta[DIM:DIM + n]
        self.ak = self.ak + delta[DIM + n:]

    def estimate(self):
        return estimate_from_state(self.F, self.b, self.ac, self.ak)

    def physical_covariance(self):
        n = self.n_anchor_clocks
        J = physical_jacobian(self.F, self.b, n)[:17 + 2 * n]
        return J @ self.P @ J.T

    def native_error(self, truth: Estimate):
        """eps = log(X X_hat^-1) for the true state, and its covariance, without the
        three coordinates of the virtual bias b_nu; anchor clock errors are appended.

        b_nu is exactly zero in truth and tightly constrained, so second-order terms
        of the error map dominate its NEES; leaving it out gives the coordinates
        that correspond to the physical state.
        """
        F_true = frame_from_estimate(truth)
        b_true = np.r_[truth.gyro_bias, truth.accel_bias, 0.0, 0.0, 0.0, truth.rho]
        X_true = (F_true, -gal3.Ad(F_true) @ b_true)
        X_hat_inv = (gal3.inverse(self.F), self.b)  # inverse of (F, -Ad(F) b)
        eps = gal3.g_log(gal3.g_compose(X_true, X_hat_inv))
        idx = np.r_[_PHYSICAL, DIM:DIM + 2 * self.n_anchor_clocks]
        anchors = np.r_[truth.anchor_offset - self.ac, truth.anchor_skew - self.ak]
        return np.r_[eps[_PHYSICAL], anchors], self.P[np.ix_(idx, idx)]
