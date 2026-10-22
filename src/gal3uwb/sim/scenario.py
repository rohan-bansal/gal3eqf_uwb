"""One simulated flight: trajectory, tag clock, IMU, UWB, and ground truth."""

from dataclasses import dataclass, field

import numpy as np

from gal3uwb.params import C, ClockNoise, ImuNoise
from gal3uwb.sim.clocks import TagClock
from gal3uwb.sim.imu import ImuData, simulate_imu
from gal3uwb.sim.trajectory import CircleTrajectory
from gal3uwb.sim.uwb import UwbData, box_anchors, simulate_uwb

_CLOCK_MARGIN = 1.0  # extra tag time simulated past the end, for late receptions [s]


@dataclass(frozen=True)
class SimConfig:
    speed: float = 0.0  # [m/s]
    skew: float = 20e-6  # tag clock skew magnitude, random sign [-]
    duration: float = 30.0  # [s]
    imu_rate: float = 200.0  # [Hz]
    uwb_rate: float = 100.0  # aggregate over all anchors [Hz]
    sigma_range: float = 0.1  # UWB timestamp noise, in metres [m]
    radius: float = 2.0  # [m]
    height: float = 1.0  # [m]
    max_offset: float = 1e-3  # initial tag clock offset ~ U(-max, max) [s]
    gyro_bias_sigma: float = 0.01  # [rad/s]
    accel_bias_sigma: float = 0.05  # [m/s^2]
    imu_noise: ImuNoise = field(default_factory=ImuNoise)
    clock_noise: ClockNoise = field(default_factory=ClockNoise)


@dataclass(frozen=True)
class Truth:
    """Ground truth at the IMU sample instants T[k] (tag time)."""

    T: np.ndarray
    t: np.ndarray  # true time of each sample
    R: np.ndarray  # (K, 3, 3)
    v: np.ndarray
    p: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    beta: np.ndarray  # C * offset [m]
    gamma: np.ndarray  # C * skew [m/s]


@dataclass(frozen=True)
class SimData:
    config: SimConfig
    anchors: np.ndarray
    imu: ImuData
    uwb: UwbData
    truth: Truth


def simulate(config: SimConfig, seed: int) -> SimData:
    rng = np.random.default_rng(seed)
    dT = 1.0 / config.imu_rate
    n = int(round(config.duration * config.imu_rate))

    traj = CircleTrajectory(config.speed, config.radius, config.height)
    skew0 = config.skew * rng.choice([-1.0, 1.0])
    offset0 = rng.uniform(-config.max_offset, config.max_offset)
    n_clock = n + int(_CLOCK_MARGIN * config.imu_rate)
    clock = TagClock.simulate(n_clock, dT, offset0, skew0, config.clock_noise, rng)

    imu, gyro_bias, accel_bias = simulate_imu(
        traj,
        clock,
        n,
        dT,
        config.imu_noise,
        rng.normal(0.0, config.gyro_bias_sigma, 3),
        rng.normal(0.0, config.accel_bias_sigma, 3),
        rng,
    )
    anchors = box_anchors()
    uwb = simulate_uwb(traj, clock, anchors, config.uwb_rate, config.duration, config.sigma_range, rng)

    t = clock.t[:n]
    truth = Truth(
        T=imu.T,
        t=t,
        R=traj.rotation(t),
        v=traj.velocity(t),
        p=traj.position(t),
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        beta=C * clock.tau[:n],
        gamma=C * clock.skew[:n],
    )
    return SimData(config, anchors, imu, uwb, truth)
