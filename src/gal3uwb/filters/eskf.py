"""Error-state EKF for IMU + listen-only UWB with tag clock offset and skew.

State: attitude R, velocity v, position p, gyro and accel biases, clock offset
beta = C * tau [m] and clock skew gamma = C * dtau/dT [m/s]. The filter runs on
the tag clock: every IMU sample and UWB reception is stamped in tag time T.

Error convention: R = R_hat Exp(dtheta), everything else additive, ordered
[dtheta, dv, dp, dbg, dba, dbeta, dgamma].

`coupled` is the only difference between the two variants:
  True:  the IMU is integrated over true time h = dT (1 - gamma / C). This is
         the Galilean time-rate coupling (the b_rho slot in Delama et al.).
  False: the IMU is integrated over tag time dT, so the clock states only enter
         the range model, as when clock states are added beside a pose filter.

Because the state is indexed by the tag clock, the offset enters only the range
model in both variants. An offset-velocity (v * dtau) term would appear only if
the state were referenced to network time instead.
"""

from dataclasses import dataclass

import numpy as np

from gal3uwb import lie
from gal3uwb.params import C, GRAVITY, ClockNoise, ImuNoise

ROT, VEL, POS, BG, BA = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
BETA, GAMMA = 15, 16
DIM = 17
_I3 = np.eye(3)


@dataclass(frozen=True)
class State:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    bg: np.ndarray
    ba: np.ndarray
    beta: float
    gamma: float

    def retract(self, d):
        return State(
            self.R @ lie.exp(d[ROT]),
            self.v + d[VEL],
            self.p + d[POS],
            self.bg + d[BG],
            self.ba + d[BA],
            self.beta + d[BETA],
            self.gamma + d[GAMMA],
        )

    def local(self, other):
        """Error d such that other = self.retract(d)."""
        d = np.empty(DIM)
        d[ROT] = lie.log(self.R.T @ other.R)
        d[VEL] = other.v - self.v
        d[POS] = other.p - self.p
        d[BG] = other.bg - self.bg
        d[BA] = other.ba - self.ba
        d[BETA] = other.beta - self.beta
        d[GAMMA] = other.gamma - self.gamma
        return d


def propagate(x: State, gyro, accel, dT, coupled):
    """Integrate one zero-order-hold IMU sample over tag time dT. Returns (x', F)."""
    w = gyro - x.bg
    f = accel - x.ba
    h = dT * (1.0 - x.gamma / C) if coupled else dT
    phi = w * h
    dR = lie.exp(phi)
    G1, G2 = lie.gamma1(phi), lie.gamma2(phi)
    dv_body = G1 @ f * h
    dp_body = G2 @ f * (h * h)

    x_new = State(
        x.R @ dR,
        x.v + x.R @ dv_body + GRAVITY * h,
        x.p + x.v * h + x.R @ dp_body + 0.5 * GRAVITY * h * h,
        x.bg,
        x.ba,
        x.beta + x.gamma * dT,
        x.gamma,
    )

    Jr = G1.T  # right Jacobian of SO(3) at phi
    F = np.eye(DIM)
    F[ROT, ROT] = dR.T
    F[ROT, BG] = -Jr * h
    F[VEL, ROT] = -x.R @ lie.hat(dv_body)
    F[VEL, BG] = 0.5 * h * h * (x.R @ lie.hat(f))
    F[VEL, BA] = -h * (x.R @ G1)
    F[POS, ROT] = -x.R @ lie.hat(dp_body)
    F[POS, VEL] = h * _I3
    F[POS, BG] = h**3 / 6.0 * (x.R @ lie.hat(f))
    F[POS, BA] = -h * h * (x.R @ G2)
    F[BETA, GAMMA] = dT
    if coupled:
        dh = -dT / C  # dh / dgamma
        F[ROT, GAMMA] = dh * (Jr @ w)
        F[VEL, GAMMA] = dh * (x_new.R @ f + GRAVITY)
        F[POS, GAMMA] = dh * x_new.v
    return x_new, F


class ProcessNoise:
    """Discrete process noise Q(dT) = Q1 dT + Q2 dT^2 + Q3 dT^3."""

    def __init__(self, imu: ImuNoise, clock: ClockNoise):
        qa = imu.accel_nd**2
        self.Q1, self.Q2, self.Q3 = np.zeros((DIM, DIM)), np.zeros((DIM, DIM)), np.zeros((DIM, DIM))
        self.Q1[ROT, ROT] = imu.gyro_nd**2 * _I3
        self.Q1[VEL, VEL] = qa * _I3
        self.Q1[BG, BG] = imu.gyro_bias_rw**2 * _I3
        self.Q1[BA, BA] = imu.accel_bias_rw**2 * _I3
        self.Q1[BETA, BETA] = C**2 * clock.q_offset
        self.Q1[GAMMA, GAMMA] = C**2 * clock.q_skew
        self.Q2[VEL, POS] = self.Q2[POS, VEL] = 0.5 * qa * _I3
        self.Q3[POS, POS] = qa / 3.0 * _I3

    def __call__(self, dT):
        return dT * (self.Q1 + dT * (self.Q2 + dT * self.Q3))


def range_model(x: State, anchor):
    """Predicted C * (T_rx - t_tx) = |p - anchor| + beta, and its Jacobian."""
    d = x.p - anchor
    r = np.sqrt(d @ d)
    H = np.zeros(DIM)
    H[POS] = d / r
    H[BETA] = 1.0
    return r + x.beta, H


class ClockEskf:
    def __init__(self, x0: State, P0, imu_noise: ImuNoise, clock_noise: ClockNoise, coupled: bool):
        self.x = x0
        self.P = np.array(P0, dtype=float)
        self.coupled = coupled
        self._Q = ProcessNoise(imu_noise, clock_noise)

    def propagate(self, gyro, accel, dT):
        self.x, F = propagate(self.x, gyro, accel, dT, self.coupled)
        self.P = F @ self.P @ F.T + self._Q(dT)

    def update_range(self, z, anchor, sigma):
        zhat, H = range_model(self.x, anchor)
        PHt = self.P @ H
        K = PHt / (H @ PHt + sigma**2)
        self.x = self.x.retract(K * (z - zhat))
        IKH = np.eye(DIM) - np.outer(K, H)
        P = IKH @ self.P @ IKH.T + sigma**2 * np.outer(K, K)
        self.P = 0.5 * (P + P.T)
