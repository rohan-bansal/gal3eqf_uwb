"""State, error and consistency helpers shared by the EKF and the EqF.

Both filters are compared in the same physical error coordinates (17 + 2n):
    [theta (body frame, R = R_hat Exp(theta)), dv, dp, d gyro bias, d accel bias, d delta, d rho,
     d Delta_1..n, d kappa_1..n]
where delta = network (anchor 0) minus tag time [s], rho = -d(delta)/dT is the
time-rate bias, and with free-running anchors Delta_i = A_i - A_0 with
kappa_i = d(Delta_i)/dT for anchors 1..n (n = N - 1; n = 0 when synchronized).
"""

from dataclasses import dataclass, field

import numpy as np

from gal3uwb import lie
from gal3uwb.params import ClockNoise, ImuNoise

ROT, VEL, POS, BG, BA = (slice(3 * i, 3 * i + 3) for i in range(5))
DELTA, RHO = 15, 16
CLOCK = slice(15, 17)
DIM = 17  # navigation, biases and the tag clock; anchor clocks follow


def anchor_slices(n):
    """(offsets, skews) slices of the n anchor clocks in the error vector."""
    return slice(DIM, DIM + n), slice(DIM + n, DIM + 2 * n)


@dataclass(frozen=True)
class Estimate:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    delta: float
    rho: float
    anchor_offset: np.ndarray = field(default_factory=lambda: np.zeros(0))  # Delta_1..n [s]
    anchor_skew: np.ndarray = field(default_factory=lambda: np.zeros(0))  # kappa_1..n [-]

    @property
    def n_anchor_clocks(self):
        return len(self.anchor_offset)

    @property
    def dim(self):
        return DIM + 2 * self.n_anchor_clocks

    def retract(self, e):
        """Apply a physical error vector: the state that lies e away from self."""
        off, skew = anchor_slices(self.n_anchor_clocks)
        return Estimate(
            self.R @ lie.exp(e[ROT]), self.v + e[VEL], self.p + e[POS], self.gyro_bias + e[BG],
            self.accel_bias + e[BA], self.delta + e[DELTA], self.rho + e[RHO], self.anchor_offset + e[off],
            self.anchor_skew + e[skew],
        )

    def error_to(self, truth):
        """Physical error vector e with truth = self.retract(e)."""
        return np.concatenate([
            lie.log(self.R.T @ truth.R), truth.v - self.v, truth.p - self.p, truth.gyro_bias - self.gyro_bias,
            truth.accel_bias - self.accel_bias, [truth.delta - self.delta, truth.rho - self.rho],
            truth.anchor_offset - self.anchor_offset, truth.anchor_skew - self.anchor_skew,
        ])


@dataclass(frozen=True)
class FilterNoise:
    """Continuous-time noise model assumed by the filters."""

    imu: ImuNoise
    clock: ClockNoise
    sigma_range: float
    # The EqF's virtual velocity-input bias b_nu is zero in truth, but in the EqF's
    # coordinates it absorbs second-order terms such as (position error) x (gyro bias
    # error), so its prior must not be much tighter than that product.
    nu_sigma0: float = 1e-4  # prior std [m/s]
    nu_nd: float = 1e-6  # virtual velocity-input noise [m/s/sqrt(Hz)]
    nu_rw: float = 1e-6  # random walk [m/s/sqrt(s)]


def clock_noise_covariance(clock: ClockNoise, n):
    """PSD of the clock noise [w_delta, w_rho, w_Delta_1..n, w_kappa_1..n], with
    d(delta)/dT = -rho + w_delta, d(rho)/dT = w_rho, d(Delta)/dT = kappa + w_Delta,
    d(kappa)/dT = w_kappa.

    Every device clock (tag and anchors) has white offset noise q_offset and skew
    random walk q_skew. Synchronized anchors (n = 0) are an ideal reference, so only
    the tag's noise remains. Otherwise all relative clocks share anchor 0's noise:
    delta = A_0 - T and Delta_i = A_i - A_0 are correlated through A_0.
    """
    q_o, q_s = clock.q_offset, clock.q_skew
    S = np.zeros((2 + 2 * n, 2 + 2 * n))
    if n == 0:
        S[0, 0], S[1, 1] = q_o, q_s
        return S
    off = np.r_[0, 2:2 + n]  # [delta, Delta_1..n]
    skew = np.r_[1, 2 + n:2 + 2 * n]  # [rho, kappa_1..n]
    # Each state is its own device's noise plus a shared anchor-0 term with coefficient c:
    # delta = o_0 - tau, Delta_i = o_i - o_0;  rho = s_tag - s_0, kappa_i = s_i - s_0.
    c_off = np.r_[1.0, -np.ones(n)]
    c_skew = -np.ones(n + 1)
    S[np.ix_(off, off)] = q_o * (np.eye(n + 1) + np.outer(c_off, c_off))
    S[np.ix_(skew, skew)] = q_s * (np.eye(n + 1) + np.outer(c_skew, c_skew))
    return S


def nees(e, P):
    """e^T P^-1 e, solved in coordinates scaled by sqrt(diag P).

    Clock offset variances are ~1e-18 s^2 next to ~1e-2 m^2 for position, so the
    covariance is badly conditioned in raw units but well conditioned once scaled.
    """
    d = np.sqrt(np.diag(P))
    Ps = P / np.outer(d, d)
    es = e / d
    return float(es @ np.linalg.solve(Ps, es))
