"""Error-state EKF on SE2(3) with IMU biases and tag clock states.

The baseline in the style of Shalaby et al. and Delama et al.'s "EKF (online)":
extended pose plus Euclidean bias and clock states, with no coupling between
the spatial and temporal errors. Attitude error is right-perturbed,
R = R_hat Exp(theta); everything else is additive. The filter runs in tag time
and the clock follows d(delta)/dT = -rho; with free-running anchors it also
carries Delta_i, kappa_i for anchors 1..N-1 with d(Delta_i)/dT = kappa_i.
"""

import numpy as np

from gal3uwb import lie
from gal3uwb.filters.common import (BA, BG, DELTA, DIM, POS, RHO, ROT, VEL, Estimate, FilterNoise, anchor_slices,
                                    clock_noise_covariance)
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
        x.anchor_offset + x.anchor_skew * dT,
        x.anchor_skew,
    )
    F = np.eye(x.dim)
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
    off, skew = anchor_slices(x.n_anchor_clocks)
    F[off, skew] = dT * np.eye(x.n_anchor_clocks)
    F_input = F[:, 9:15].copy()
    F_input[9:15] = 0.0  # white input noise does not persist in the bias states
    return x_new, F, F_input


def anchor_offsets(x: Estimate, n_anchors):
    """A_i - A_0 for every anchor: zero for anchor 0, and for all when synchronized."""
    return np.r_[0.0, x.anchor_offset] if x.n_anchor_clocks else np.zeros(n_anchors)


def anchor_columns(x: Estimate, n_anchors):
    """State column of each anchor's offset Delta_i, or None for anchor 0 / synchronized."""
    if not x.n_anchor_clocks:
        return [None] * n_anchors
    return [None] + [DIM + i for i in range(x.n_anchor_clocks)]


def range_model(x: Estimate, anchors, lever_arm):
    """Predicted y = C (delta + Delta_i) + |p + R l - a_i| for each anchor, and its Jacobian."""
    p_tag = x.p + x.R @ lever_arm
    diff = p_tag[None, :] - anchors
    r = np.linalg.norm(diff, axis=1)
    u = diff / r[:, None]
    H = np.zeros((len(anchors), x.dim))
    H[:, ROT] = -u @ x.R @ lie.hat(lever_arm)
    H[:, POS] = u
    H[:, DELTA] = C
    for i, col in enumerate(anchor_columns(x, len(anchors))):
        if col is not None:
            H[i, col] = C
    return C * (x.delta + anchor_offsets(x, len(anchors))) + r, H


def sync_model(x_dim, offsets, columns, tx, anchors):
    """Predicted y_ij = |a_j - a_tx| + C (Delta_j - Delta_tx) at every other anchor j,
    its Jacobian (x_dim columns) and the receiving anchors. Shared with the EqF,
    whose anchor clocks sit at different columns."""
    rx = np.array([j for j in range(len(anchors)) if j != tx])
    h = np.linalg.norm(anchors[rx] - anchors[tx], axis=1) + C * (offsets[rx] - offsets[tx])
    H = np.zeros((len(rx), x_dim))
    for row, j in enumerate(rx):
        if columns[j] is not None:
            H[row, columns[j]] += C
        if columns[tx] is not None:
            H[row, columns[tx]] -= C
    return h, H, rx


class Ekf:
    name = "EKF"

    def __init__(self, x0: Estimate, P0, noise: FilterNoise):
        self.x, self.P, self.noise = x0, P0.copy(), noise
        imu, n = noise.imu, x0.n_anchor_clocks
        self.q_input = np.diag(np.r_[np.full(3, imu.gyro_nd**2), np.full(3, imu.accel_nd**2)])
        self.q_walk = np.zeros(x0.dim)
        self.q_walk[BG] = imu.gyro_bias_rw**2
        self.q_walk[BA] = imu.accel_bias_rw**2
        self.clock_idx = np.r_[DELTA, RHO, DIM:DIM + 2 * n]
        self.q_clock = clock_noise_covariance(noise.clock, n)

    def propagate(self, gyro, accel, dT):
        self.x, F, F_input = propagate(self.x, gyro, accel, dT)
        Q = F_input @ self.q_input @ F_input.T / dT + np.diag(self.q_walk * dT)
        Q[np.ix_(self.clock_idx, self.clock_idx)] += self.q_clock * dT
        self.P = F @ self.P @ F.T + Q

    def update(self, y, anchors, lever_arm):
        h, H = range_model(self.x, anchors, lever_arm)
        self._correct(y - h, H)

    def update_sync(self, y, tx, anchors):
        """Anchor-to-anchor packet from anchor tx; y holds every anchor's timestamp (NaN at tx)."""
        n = len(anchors)
        h, H, rx = sync_model(self.x.dim, anchor_offsets(self.x, n), anchor_columns(self.x, n), tx, anchors)
        self._correct(y[rx] - h, H)

    def _correct(self, r, H):
        dim = self.x.dim
        R = self.noise.sigma_range**2 * np.eye(len(r))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ r
        I_KH = np.eye(dim) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        G = np.eye(dim)
        G[ROT, ROT] -= 0.5 * lie.hat(dx[ROT])  # covariance reset for the attitude injection
        self.P = G @ (0.5 * (P + P.T)) @ G.T
        self.x = self.x.retract(dx)

    def estimate(self):
        return self.x

    def physical_covariance(self):
        return self.P

    def native_error(self, truth: Estimate):
        return self.x.error_to(truth), self.P
