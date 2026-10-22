"""Error-state EKF on SE2(3) with IMU biases and tag clock states.

The baseline in the style of Shalaby et al. and Delama et al.'s "EKF (online)":
extended pose plus Euclidean bias and clock states, with no coupling between
the spatial and temporal errors. Attitude error is right-perturbed,
R = R_hat Exp(theta); everything else is additive. The filter runs in tag time
and the clock follows d(delta)/dT = -rho.
"""

import numpy as np

from gal3uwb import lie
from gal3uwb.filters.common import BA, BG, DELTA, DIM, POS, RHO, ROT, VEL, Estimate, FilterNoise
from gal3uwb.params import C, GRAVITY


def propagate(x: Estimate, gyro, accel, dT):
    """Exact zero-order-hold step; returns (x_new, F, F_input) where F_input maps a
    constant gyro/accel offset over the step to the error (used for the noise)."""
    w, f = gyro - x.gyro_bias, accel - x.accel_bias
    phi = w * dT
    dR = lie.exp(phi)
    G1, G2 = lie.gamma1(phi), lie.gamma2(phi)
    dv_body, dp_body = G1 @ f * dT, G2 @ f * dT**2
    R = x.R
    x_new = Estimate(
        R @ dR,
        x.v + R @ dv_body + GRAVITY * dT,
        x.p + x.v * dT + R @ dp_body + 0.5 * GRAVITY * dT**2,
        x.gyro_bias,
        x.accel_bias,
        x.delta - x.rho * dT,
        x.rho,
    )
    F = np.eye(DIM)
    F[ROT, ROT] = dR.T
    F[ROT, BG] = -G1.T * dT  # right Jacobian of SO(3)
    F[VEL, ROT] = -R @ lie.hat(dv_body)
    F[VEL, BG] = 0.5 * dT**2 * R @ lie.hat(f)
    F[VEL, BA] = -dT * R @ G1
    F[POS, ROT] = -R @ lie.hat(dp_body)
    F[POS, VEL] = dT * np.eye(3)
    F[POS, BG] = dT**3 / 6.0 * R @ lie.hat(f)
    F[POS, BA] = -(dT**2) * R @ G2
    F[DELTA, RHO] = -dT
    F_input = F[:, 9:15].copy()
    F_input[9:15] = 0.0  # white input noise does not persist in the bias states
    return x_new, F, F_input


def range_model(x: Estimate, anchors, lever_arm):
    """Predicted y = C delta + |p + R l - a| for each anchor, and its Jacobian."""
    p_tag = x.p + x.R @ lever_arm
    diff = p_tag[None, :] - anchors
    r = np.linalg.norm(diff, axis=1)
    u = diff / r[:, None]
    H = np.zeros((len(anchors), DIM))
    H[:, ROT] = -u @ x.R @ lie.hat(lever_arm)
    H[:, POS] = u
    H[:, DELTA] = C
    return C * x.delta + r, H


class Ekf:
    name = "EKF"

    def __init__(self, x0: Estimate, P0, noise: FilterNoise):
        self.x, self.P, self.noise = x0, P0.copy(), noise
        imu, clock = noise.imu, noise.clock
        self.q_input = np.diag(np.r_[np.full(3, imu.gyro_nd**2), np.full(3, imu.accel_nd**2)])
        self.q_walk = np.zeros(DIM)
        self.q_walk[BG] = imu.gyro_bias_rw**2
        self.q_walk[BA] = imu.accel_bias_rw**2
        self.q_walk[DELTA] = clock.q_offset
        self.q_walk[RHO] = clock.q_skew

    def propagate(self, gyro, accel, dT):
        self.x, F, F_input = propagate(self.x, gyro, accel, dT)
        Q = F_input @ self.q_input @ F_input.T / dT + np.diag(self.q_walk * dT)
        self.P = F @ self.P @ F.T + Q

    def update(self, y, anchors, lever_arm):
        h, H = range_model(self.x, anchors, lever_arm)
        R = self.noise.sigma_range**2 * np.eye(len(y))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ (y - h)
        I_KH = np.eye(DIM) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        G = np.eye(DIM)
        G[ROT, ROT] -= 0.5 * lie.hat(dx[ROT])  # covariance reset for the attitude injection
        self.P = G @ (0.5 * (P + P.T)) @ G.T
        self.x = self.x.retract(dx)

    def estimate(self):
        return self.x

    def physical_covariance(self):
        return self.P

    def native_error(self, truth: Estimate):
        return self.x.error_to(truth), self.P
