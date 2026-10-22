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
"""

import numpy as np

from gal3uwb import gal3, lie
from gal3uwb.filters.common import BA, BG, DELTA, POS, RHO, ROT, VEL, Estimate, FilterNoise
from gal3uwb.params import C, GRAVITY

N = gal3.DIM  # 10
DIM = 2 * N  # 20
G_N = np.r_[0.0, 0.0, 0.0, -GRAVITY, 0.0, 0.0, 0.0, 1.0]
_PHYSICAL = np.r_[0:16, 19]  # eps without the b_nu coordinates


def frame_from_estimate(x: Estimate):
    d = x.delta
    return gal3.make(x.R, x.v - GRAVITY * d, x.p - 0.5 * GRAVITY * d * d, d)


def estimate_from_state(F, b):
    R, v_F, p_F, d = gal3.parts(F)
    return Estimate(R.copy(), v_F + GRAVITY * d, p_F + 0.5 * GRAVITY * d * d, b[0:3].copy(), b[3:6].copy(), d,
                    b[9])


def physical_jacobian(F, b):
    """d(physical error)/d(eps) at the estimate, 20x20.

    Rows are the 17 physical error coordinates (common.py) followed by the
    virtual bias b_nu; columns are eps = (Gal(3) error, bias error).
    """
    R, v_F, p_F, d = gal3.parts(F)
    J = np.zeros((DIM, DIM))
    J[ROT, 0:3] = R.T
    J[VEL, 0:3] = -lie.hat(v_F)
    J[VEL, 3:6] = np.eye(3)
    J[VEL, 9] = GRAVITY
    J[POS, 0:3] = -lie.hat(p_F)
    J[POS, 3:6] = d * np.eye(3)
    J[POS, 6:9] = np.eye(3)
    J[POS, 9] = GRAVITY * d
    J[DELTA, 9] = 1.0
    # Bias error: eps_b = -Ad(F) (b - b_hat) to first order.
    dbias = -gal3.Ad(gal3.inverse(F))
    J[BG, N:] = dbias[0:3]
    J[BA, N:] = dbias[3:6]
    J[RHO, N:] = dbias[9]
    J[17:20, N:] = dbias[6:9]
    return J


class Eqf:
    name = "EqF"

    def __init__(self, x0: Estimate, P0, noise: FilterNoise):
        """P0 is the 17x17 covariance in physical coordinates; it is mapped to eps."""
        self.F = frame_from_estimate(x0)
        self.b = np.r_[x0.gyro_bias, x0.accel_bias, 0.0, 0.0, 0.0, x0.rho]
        self.noise = noise
        P_phys = np.zeros((DIM, DIM))
        P_phys[:17, :17] = P0
        P_phys[17:, 17:] = noise.nu_sigma0**2 * np.eye(3)
        J_inv = np.linalg.inv(physical_jacobian(self.F, self.b))
        self.P = J_inv @ P_phys @ J_inv.T
        imu, clock = noise.imu, noise.clock
        # Input noise on w_N = (omega, a, nu, rho) and random walk on b, in gal(3) order.
        self.q_input = np.r_[np.full(3, imu.gyro_nd**2), np.full(3, imu.accel_nd**2), np.full(3, noise.nu_nd**2),
                             clock.q_offset]
        self.q_walk = np.r_[np.full(3, imu.gyro_bias_rw**2), np.full(3, imu.accel_bias_rw**2),
                            np.full(3, noise.nu_rw**2), clock.q_skew]

    def propagate(self, gyro, accel, dT):
        F, b = self.F, self.b
        u = np.r_[gyro, accel, 0.0, 0.0, 0.0, 1.0] - b
        Ad_F = gal3.Ad(F)
        w_ring = Ad_F @ u * dT
        E_g = gal3.exp(-G_N * dT)
        A1 = gal3.Ad(E_g)
        J = gal3.left_jacobian(w_ring)
        A2 = A1 @ gal3.Ad(gal3.exp(w_ring))
        A = np.zeros((DIM, DIM))
        A[:N, :N] = A1
        A[:N, N:] = A1 @ J * dT
        A[N:, N:] = A2
        B_input = -A1 @ J @ Ad_F * dT
        B_walk = -A2 @ Ad_F * dT
        Q = np.zeros((DIM, DIM))
        Q[:N, :N] = (B_input * self.q_input) @ B_input.T / dT
        Q[N:, N:] = (B_walk * self.q_walk) @ B_walk.T / dT
        self.P = A @ self.P @ A.T + Q
        self.F = E_g @ F @ gal3.exp(u * dT)

    def range_model(self, anchors, lever_arm):
        R, v_F, p_F, d = gal3.parts(self.F)
        p_tag = p_F + 0.5 * GRAVITY * d * d + R @ lever_arm
        diff = p_tag[None, :] - anchors
        r = np.linalg.norm(diff, axis=1)
        u = diff / r[:, None]
        H = np.zeros((len(anchors), DIM))
        H[:, 0:3] = -u @ lie.hat(p_F + R @ lever_arm)
        H[:, 3:6] = d * u
        H[:, 6:9] = u
        H[:, 9] = C + d * (u @ GRAVITY)
        return C * d + r, H

    def update(self, y, anchors, lever_arm):
        h, H = self.range_model(anchors, lever_arm)
        R = self.noise.sigma_range**2 * np.eye(len(y))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        delta = K @ (y - h)
        I_KH = np.eye(DIM) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        J = gal3.g_left_jacobian(delta)  # reset (Ge et al.)
        self.P = J @ (0.5 * (P + P.T)) @ J.T
        X_hat = (self.F, -gal3.Ad(self.F) @ self.b)
        self.F, X_b = gal3.g_compose(gal3.g_exp(delta), X_hat)
        self.b = -gal3.Ad(gal3.inverse(self.F)) @ X_b

    def estimate(self):
        return estimate_from_state(self.F, self.b)

    def physical_covariance(self):
        J = physical_jacobian(self.F, self.b)[:17]
        return J @ self.P @ J.T

    def native_error(self, truth: Estimate):
        """eps = log(X X_hat^-1) for the true state, and its covariance, without the
        three coordinates of the virtual bias b_nu.

        b_nu is exactly zero in truth and tightly constrained, so second-order terms
        of the error map dominate its NEES; leaving it out gives the 17 coordinates
        that correspond to the physical state.
        """
        F_true = frame_from_estimate(truth)
        b_true = np.r_[truth.gyro_bias, truth.accel_bias, 0.0, 0.0, 0.0, truth.rho]
        X_true = (F_true, -gal3.Ad(F_true) @ b_true)
        X_hat_inv = (gal3.inverse(self.F), self.b)  # inverse of (F, -Ad(F) b)
        eps = gal3.g_log(gal3.g_compose(X_true, X_hat_inv))
        return eps[_PHYSICAL], self.P[np.ix_(_PHYSICAL, _PHYSICAL)]
