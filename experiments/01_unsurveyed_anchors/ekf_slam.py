"""Error-state EKF with the anchor positions as extra additive states.

The navigation part is the baseline EKF of experiment 00 (gal3uwb.filters.ekf):
attitude error multiplied in on the right, R = R_hat Exp(theta), every other
error added. Anchor positions are static, with additive errors a = a_hat + e_a.

Neither the ranges nor the anchor-to-anchor packets change when the whole scene
(drone and anchors) is rotated about gravity or translated, so that motion is
unobservable. In these coordinates the rotated direction depends on the current
estimates of v, p and the anchors, which move with every update; this is how
an EKF-SLAM gains information it does not have.
"""

import numpy as np

from gal3uwb import lie
from gal3uwb.filters.common import BA, BG, DELTA, DIM, POS, RHO, ROT, clock_noise_covariance
from gal3uwb.filters.ekf import anchor_columns, anchor_offsets, propagate
from gal3uwb.params import C

from model import SlamEstimate


class EkfSlam:
    name = "EKF"
    reset_attitude = True

    def __init__(self, x0: SlamEstimate, P0, noise):
        self.x, self.P, self.noise = x0, P0.copy(), noise
        imu, n = noise.imu, x0.nav.n_anchor_clocks
        self.k = x0.nav.dim  # first anchor-position column
        self.q_input = np.diag(np.r_[np.full(3, imu.gyro_nd**2), np.full(3, imu.accel_nd**2)])
        self.q_walk = np.zeros(self.k)
        self.q_walk[BG] = imu.gyro_bias_rw**2
        self.q_walk[BA] = imu.accel_bias_rw**2
        self.clock_idx = np.r_[DELTA, RHO, DIM:DIM + 2 * n]
        self.q_clock = clock_noise_covariance(noise.clock, n)

    def propagate(self, gyro, accel, dT):
        """The anchors are static, so only the navigation rows of P change."""
        nav, F, F_input = self.transition(gyro, accel, dT)
        k = self.k
        Q = F_input @ self.q_input @ F_input.T / dT + np.diag(self.q_walk * dT)
        Q[np.ix_(self.clock_idx, self.clock_idx)] += self.q_clock * dT
        P = self.P
        P[:k, :k] = F @ P[:k, :k] @ F.T + Q
        P[:k, k:] = F @ P[:k, k:]
        P[k:, :k] = P[:k, k:].T
        self.x = SlamEstimate(nav, self.x.anchors)

    def transition(self, gyro, accel, dT):
        """Propagated navigation state and its Jacobian, at the current estimate."""
        return propagate(self.x.nav, gyro, accel, dT)

    def jacobian_point(self):
        """(navigation state, anchors) at which the measurement Jacobians are evaluated."""
        return self.x.nav, self.x.anchors

    def range_model(self, lever_arm):
        """Tag packet: y_i = C (delta + Delta_i) + |p + R l - a_i| at every anchor."""
        nav, A = self.x.nav, self.x.anchors
        r = np.linalg.norm(nav.p + nav.R @ lever_arm - A, axis=1)
        h = C * (nav.delta + anchor_offsets(nav, len(A))) + r
        nav, A = self.jacobian_point()
        diff = nav.p + nav.R @ lever_arm - A
        u = diff / np.linalg.norm(diff, axis=1)[:, None]
        k = self.k
        H = np.zeros((len(A), self.x.dim))
        H[:, ROT] = -u @ nav.R @ lie.hat(lever_arm)
        H[:, POS] = u
        H[:, DELTA] = C
        for i, col in enumerate(anchor_columns(nav, len(A))):
            if col is not None:
                H[i, col] = C
            H[i, k + 3 * i:k + 3 * i + 3] = -u[i]
        return h, H

    def sync_model(self, tx):
        """Anchor packet from anchor tx: y_j = C (Delta_j - Delta_tx) + |a_j - a_tx|."""
        nav, A = self.x.nav, self.x.anchors
        rx = np.array([j for j in range(len(A)) if j != tx])
        offsets, cols = anchor_offsets(nav, len(A)), anchor_columns(nav, len(A))
        h = np.linalg.norm(A[rx] - A[tx], axis=1) + C * (offsets[rx] - offsets[tx])
        _, A = self.jacobian_point()
        d = A[rx] - A[tx]
        u = d / np.linalg.norm(d, axis=1)[:, None]
        k = self.k
        H = np.zeros((len(rx), self.x.dim))
        for row, j in enumerate(rx):
            if cols[j] is not None:
                H[row, cols[j]] += C
            if cols[tx] is not None:
                H[row, cols[tx]] -= C
            H[row, k + 3 * j:k + 3 * j + 3] += u[row]
            H[row, k + 3 * tx:k + 3 * tx + 3] -= u[row]
        return h, H, rx

    def update(self, y, lever_arm):
        h, H = self.range_model(lever_arm)
        self._correct(y - h, H)

    def update_sync(self, y, tx):
        h, H, rx = self.sync_model(tx)
        self._correct(y[rx] - h, H)

    def _correct(self, r, H):
        dim = self.x.dim
        R = self.noise.sigma_range**2 * np.eye(len(r))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ r
        I_KH = np.eye(dim) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        P = 0.5 * (P + P.T)
        if self.reset_attitude:  # covariance reset for the attitude injection
            G = np.eye(dim)
            G[ROT, ROT] -= 0.5 * lie.hat(dx[ROT])
            P = G @ P @ G.T
        self.P = P
        self.x = self.x.retract(dx)

    def estimate(self):
        return self.x

    def physical_covariance(self):
        return self.P

    def native_error(self, truth: SlamEstimate):
        return self.x.error_to(truth), self.P
