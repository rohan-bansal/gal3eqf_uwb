"""State, error and consistency helpers shared by the EKF and the EqF.

Both filters are compared in the same physical error coordinates (17-dim):
    [theta (body frame, R = R_hat Exp(theta)), dv, dp, d gyro bias, d accel bias, d delta, d rho]
where delta = network minus tag time [s] and rho is the time-rate bias (tag skew).
"""

from dataclasses import dataclass

import numpy as np

from gal3uwb import lie
from gal3uwb.params import ClockNoise, ImuNoise

ROT, VEL, POS, BG, BA = (slice(3 * i, 3 * i + 3) for i in range(5))
DELTA, RHO = 15, 16
CLOCK = slice(15, 17)
DIM = 17


@dataclass(frozen=True)
class Estimate:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    delta: float
    rho: float

    def retract(self, e):
        """Apply a physical error vector: the state that lies e away from self."""
        return Estimate(
            self.R @ lie.exp(e[ROT]), self.v + e[VEL], self.p + e[POS], self.gyro_bias + e[BG],
            self.accel_bias + e[BA], self.delta + e[DELTA], self.rho + e[RHO],
        )

    def error_to(self, truth):
        """Physical error vector e with truth = self.retract(e)."""
        return np.concatenate([
            lie.log(self.R.T @ truth.R), truth.v - self.v, truth.p - self.p, truth.gyro_bias - self.gyro_bias,
            truth.accel_bias - self.accel_bias, [truth.delta - self.delta, truth.rho - self.rho],
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


def nees(e, P):
    """e^T P^-1 e, solved in coordinates scaled by sqrt(diag P).

    Clock offset variances are ~1e-18 s^2 next to ~1e-2 m^2 for position, so the
    covariance is badly conditioned in raw units but well conditioned once scaled.
    """
    d = np.sqrt(np.diag(P))
    Ps = P / np.outer(d, d)
    es = e / d
    return float(es @ np.linalg.solve(Ps, es))
