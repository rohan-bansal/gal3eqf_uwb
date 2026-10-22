"""IMU clocked by the tag clock."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import ImuNoise


@dataclass(frozen=True)
class ImuData:
    """Sample k is held over tag-time interval [T[k], T[k] + dT)."""

    T: np.ndarray
    dT: float
    gyro: np.ndarray  # (K, 3) [rad/s]
    accel: np.ndarray  # (K, 3) [m/s^2]


def simulate_imu(traj, clock, n, dT, noise: ImuNoise, gyro_bias0, accel_bias0, rng):
    """Return (ImuData, gyro_bias, accel_bias); the bias arrays are ground truth.

    Each sample is the true signal at the midpoint of its hold interval, which
    makes zero-order-hold integration second-order accurate. A real IMU samples
    at the interval start; the midpoint keeps integration error out of the study.
    """
    T = np.arange(n) * dT
    t_mid = clock.true_time(T + 0.5 * dT)

    def random_walk(b0, sigma):
        steps = rng.normal(0.0, sigma * np.sqrt(dT), (n - 1, 3))
        return b0 + np.concatenate([np.zeros((1, 3)), np.cumsum(steps, axis=0)])

    gyro_bias = random_walk(gyro_bias0, noise.gyro_bias_rw)
    accel_bias = random_walk(accel_bias0, noise.accel_bias_rw)
    gyro = traj.angular_velocity(t_mid) + gyro_bias + rng.normal(0.0, noise.gyro_nd / np.sqrt(dT), (n, 3))
    accel = traj.specific_force(t_mid) + accel_bias + rng.normal(0.0, noise.accel_nd / np.sqrt(dT), (n, 3))
    return ImuData(T, dT, gyro, accel), gyro_bias, accel_bias
