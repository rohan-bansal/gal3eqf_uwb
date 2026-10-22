"""Synthetic flights for the synchronized-anchor scenarios.

Step 1 ("ideal_clock"): the tag clock has an unknown constant offset and no skew.
Step 2 ("gaussian_skew"): the tag-vs-network skew is drawn from N(0, skew_sigma^2)
per flight and then random-walks like a TCXO.
"""

from dataclasses import dataclass, field

import numpy as np

from gal3uwb.params import ClockNoise, ImuNoise
from gal3uwb.sim.clocks import TagClock
from gal3uwb.sim.imu import ImuData, simulate_imu
from gal3uwb.sim.trajectory import CircleTrajectory
from gal3uwb.sim.uwb import UwbData, box_anchors, simulate_uwb


@dataclass(frozen=True)
class ClockModel:
    skew_sigma: float  # std of the per-flight tag skew [-]
    noise: ClockNoise
    max_offset: float = 1e-3  # initial offset is uniform in +/- this [s]


SCENARIOS = {
    "ideal_clock": ClockModel(skew_sigma=0.0, noise=ClockNoise(q_offset=0.0, q_skew=0.0)),
    "gaussian_skew": ClockModel(skew_sigma=20e-6, noise=ClockNoise()),
}


@dataclass(frozen=True)
class SimConfig:
    speed: float = 0.0  # [m/s]
    clock: ClockModel = SCENARIOS["ideal_clock"]
    duration: float = 30.0  # [s]
    imu_rate: float = 200.0  # [Hz]
    uwb_rate: float = 20.0  # tag transmissions per second [Hz]
    sigma_range: float = 0.1  # per-anchor timestamp noise, in metres [m]
    radius: float = 2.0
    height: float = 1.0
    gyro_bias_sigma: float = 0.01  # [rad/s]
    accel_bias_sigma: float = 0.05  # [m/s^2]
    lever_arm: tuple = (0.0, 0.0, 0.0)  # tag position in the body frame [m]
    imu: ImuNoise = field(default_factory=ImuNoise)


@dataclass(frozen=True)
class Truth:
    """Ground truth at the IMU sample instants (tag times T)."""

    T: np.ndarray
    R: np.ndarray  # (K, 3, 3) world-from-body
    v: np.ndarray  # (K, 3)
    p: np.ndarray  # (K, 3)
    gyro_bias: np.ndarray  # (K, 3)
    accel_bias: np.ndarray  # (K, 3)
    delta: np.ndarray  # (K,) network minus tag time [s]
    rho: np.ndarray  # (K,) time-rate bias, the tag skew [-]


@dataclass(frozen=True)
class SimData:
    imu: ImuData
    uwb: UwbData
    truth: Truth
    lever_arm: np.ndarray
    imu_noise: ImuNoise
    clock_noise: ClockNoise


def truth_at(traj, clock, T, gyro_bias, accel_bias):
    t = clock.true_time(T)
    return Truth(T, traj.rotation(t), traj.velocity(t), traj.position(t), gyro_bias, accel_bias,
                 -clock.offset(T), clock.skew_at(T))


def simulate(config: SimConfig, seed: int) -> SimData:
    rng = np.random.default_rng(seed)
    dT = 1.0 / config.imu_rate
    n = int(round(config.duration * config.imu_rate)) + 1
    clock_model = config.clock
    clock = TagClock.simulate(
        n + int(config.imu_rate),  # a second of margin for transmissions near the end
        dT,
        offset0=rng.uniform(-clock_model.max_offset, clock_model.max_offset),
        skew0=rng.normal(0.0, clock_model.skew_sigma),
        noise=clock_model.noise,
        rng=rng,
    )
    traj = CircleTrajectory(config.speed, config.radius, config.height)
    T = np.arange(n) * dT
    gyro_bias0 = rng.normal(0.0, config.gyro_bias_sigma, 3)
    accel_bias0 = rng.normal(0.0, config.accel_bias_sigma, 3)
    imu, gyro_bias, accel_bias = simulate_imu(traj, clock, T, config.imu, gyro_bias0, accel_bias0, rng)
    lever_arm = np.array(config.lever_arm, dtype=float)
    T_tx = np.arange(rng.uniform(0.0, 1.0 / config.uwb_rate), T[-1], 1.0 / config.uwb_rate)
    uwb = simulate_uwb(traj, clock, T_tx, box_anchors(), lever_arm, config.sigma_range, rng)
    truth = truth_at(traj, clock, T, gyro_bias, accel_bias)
    return SimData(imu, uwb, truth, lever_arm, config.imu, clock_model.noise)
