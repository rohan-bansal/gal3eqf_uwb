"""IMU clocked by the tag clock."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import ImuNoise


@dataclass(frozen=True)
class ImuData:
    """Sample k is held over tag-time interval [T[k], T[k+1]); the last sample is unused."""

    T: np.ndarray  # (K,) tag time [s]
    gyro: np.ndarray  # (K, 3) [rad/s]
    accel: np.ndarray  # (K, 3) [m/s^2]


def simulate_imu(traj, clock, T, noise: ImuNoise, gyro_bias0, accel_bias0, rng):
    """Return (ImuData, gyro_bias, accel_bias) on tag-time grid T; the bias arrays are ground truth.

    Each sample is the true signal at the midpoint of its hold interval, which
    makes zero-order-hold integration second-order accurate. A real IMU samples
    at the interval start; the midpoint keeps integration error out of the study.
    """
    n = len(T)
    dT = np.diff(T, append=2 * T[-1] - T[-2])
    t_mid = clock.true_time(T + 0.5 * dT)

    def random_walk(b0, sigma):
        steps = rng.normal(0.0, 1.0, (n - 1, 3)) * (sigma * np.sqrt(dT[:-1]))[:, None]
        return b0 + np.concatenate([np.zeros((1, 3)), np.cumsum(steps, axis=0)])

    gyro_bias = random_walk(gyro_bias0, noise.gyro_bias_rw)
    accel_bias = random_walk(accel_bias0, noise.accel_bias_rw)
    white = rng.normal(0.0, 1.0, (2, n, 3)) / np.sqrt(dT)[None, :, None]
    gyro = traj.angular_velocity(t_mid) + gyro_bias + noise.gyro_nd * white[0]
    accel = traj.specific_force(t_mid) + accel_bias + noise.accel_nd * white[1]
    return ImuData(T, gyro, accel), gyro_bias, accel_bias
