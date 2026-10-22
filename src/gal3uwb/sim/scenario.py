"""Synthetic flights for the three clock scenarios.

Step 1 ("ideal_clock"): synchronized anchors; the tag clock has an unknown
constant offset and no skew.
Step 2 ("gaussian_skew"): synchronized anchors; the tag-vs-network skew is drawn
from N(0, skew_sigma^2) per flight and then random-walks like a TCXO.
Step 3 ("unsynced"): every anchor also runs its own TCXO clock with an arbitrary
offset and an N(0, skew_sigma^2) skew; anchors exchange packets at sync_rate.
"""

from dataclasses import dataclass, field

import numpy as np

from gal3uwb.params import ClockNoise, ImuNoise
from gal3uwb.sim.clocks import AnchorClock, TagClock
from gal3uwb.sim.imu import ImuData, simulate_imu
from gal3uwb.sim.trajectory import CircleTrajectory
from gal3uwb.sim.uwb import SyncData, UwbData, box_anchors, simulate_sync, simulate_uwb


@dataclass(frozen=True)
class ClockModel:
    skew_sigma: float  # std of the per-flight skew of each free-running clock [-]
    noise: ClockNoise
    max_offset: float = 1e-3  # initial offsets are uniform in +/- this [s]
    synced: bool = True  # anchors share true time; otherwise each has its own clock
    sync_rate: float = 10.0  # anchor-to-anchor packets per anchor per second [Hz]


SCENARIOS = {
    "ideal_clock": ClockModel(skew_sigma=0.0, noise=ClockNoise(q_offset=0.0, q_skew=0.0)),
    "gaussian_skew": ClockModel(skew_sigma=20e-6, noise=ClockNoise()),
    "unsynced": ClockModel(skew_sigma=20e-6, noise=ClockNoise(), synced=False),
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
    delta: np.ndarray  # (K,) network (anchor 0) minus tag time [s]
    rho: np.ndarray  # (K,) time-rate bias: -d(delta)/dT [-]
    anchor_offset: np.ndarray  # (K, N-1) A_i - A_0 for anchors 1..N-1, empty if synced [s]
    anchor_skew: np.ndarray  # (K, N-1) d(A_i - A_0)/dT [-]


@dataclass(frozen=True)
class SimData:
    imu: ImuData
    uwb: UwbData
    truth: Truth
    lever_arm: np.ndarray
    imu_noise: ImuNoise
    clock_noise: ClockNoise
    sync: SyncData | None = None  # anchor-to-anchor packets, unsynced anchors only


def simulate_anchor_clocks(clock_model: ClockModel, n_anchors, t_end, rng, dt=0.005):
    """Free-running anchor clocks on a true-time grid from -1 s to t_end + 2 s, or
    None for synchronized anchors."""
    if clock_model.synced:
        return None
    n = int(np.ceil((t_end + 3.0) / dt))
    return [
        AnchorClock.simulate(-1.0, n, dt, rng.uniform(-clock_model.max_offset, clock_model.max_offset),
                             rng.normal(0.0, clock_model.skew_sigma), clock_model.noise, rng)
        for _ in range(n_anchors)
    ]


def clock_truth(tag_clock, anchor_clocks, T):
    """Filter clock states at tag times T: (delta, rho, anchor_offset, anchor_skew)."""
    tau, skew = tag_clock.offset(T), tag_clock.skew_at(T)
    if anchor_clocks is None:
        empty = np.zeros((len(T), 0))
        return -tau, skew, empty, empty
    t = tag_clock.true_time(T)
    o = np.stack([c.offset_at(t) for c in anchor_clocks], -1)
    s = np.stack([c.skew_at(t) for c in anchor_clocks], -1)
    dt_dT = 1.0 - skew
    return o[:, 0] - tau, skew - s[:, 0] * dt_dT, o[:, 1:] - o[:, :1], (s[:, 1:] - s[:, :1]) * dt_dT[:, None]


def truth_at(traj, tag_clock, anchor_clocks, T, gyro_bias, accel_bias):
    t = tag_clock.true_time(T)
    return Truth(T, traj.rotation(t), traj.velocity(t), traj.position(t), gyro_bias, accel_bias,
                 *clock_truth(tag_clock, anchor_clocks, T))


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
    anchors = box_anchors()
    anchor_clocks = simulate_anchor_clocks(clock_model, len(anchors), config.duration, rng)
    traj = CircleTrajectory(config.speed, config.radius, config.height)
    T = np.arange(n) * dT
    gyro_bias0 = rng.normal(0.0, config.gyro_bias_sigma, 3)
    accel_bias0 = rng.normal(0.0, config.accel_bias_sigma, 3)
    imu, gyro_bias, accel_bias = simulate_imu(traj, clock, T, config.imu, gyro_bias0, accel_bias0, rng)
    lever_arm = np.array(config.lever_arm, dtype=float)
    T_tx = np.arange(rng.uniform(0.0, 1.0 / config.uwb_rate), T[-1], 1.0 / config.uwb_rate)
    uwb = simulate_uwb(traj, clock, T_tx, anchors, lever_arm, config.sigma_range, rng, anchor_clocks)
    sync = None
    if anchor_clocks is not None:
        sync = simulate_sync(clock, anchor_clocks, anchors, clock_model.sync_rate, (T[0], T[-1]), config.sigma_range,
                             rng)
    truth = truth_at(traj, clock, anchor_clocks, T, gyro_bias, accel_bias)
    return SimData(imu, uwb, truth, lever_arm, config.imu, clock_model.noise, sync)
