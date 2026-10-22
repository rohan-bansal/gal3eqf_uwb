from dataclasses import replace

import numpy as np

from gal3uwb.filters.eskf import propagate
from gal3uwb.eval.runner import truth_state
from gal3uwb.params import C, ClockNoise, ImuNoise
from gal3uwb.sim.scenario import SimConfig, simulate
from gal3uwb.sim.trajectory import CircleTrajectory

NOISELESS = SimConfig(
    speed=2.0,
    skew=1e-2,
    duration=10.0,
    sigma_range=0.0,
    gyro_bias_sigma=0.0,
    accel_bias_sigma=0.0,
    imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
    clock_noise=ClockNoise(0.0, 0.0),
)


def test_trajectory_kinematics_are_consistent():
    traj = CircleTrajectory(2.0)
    t, h = np.linspace(0, 5, 50), 1e-5
    np.testing.assert_allclose((traj.position(t + h) - traj.position(t - h)) / (2 * h), traj.velocity(t), atol=1e-6)
    np.testing.assert_allclose(
        (traj.velocity(t + h) - traj.velocity(t - h)) / (2 * h), traj.acceleration(t), atol=1e-6
    )
    R = traj.rotation(t)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", R, R), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)


def test_clock_maps_are_inverse():
    sim = simulate(SimConfig(skew=1e-2), seed=0)
    T = sim.truth.T
    t = sim.truth.t
    np.testing.assert_allclose(T - t, sim.truth.beta / C, atol=1e-15)


def test_uwb_timestamps_match_clock_for_negative_offset():
    # Negative offsets start the tag clock after t = 0; early packets must be dropped, not clamped.
    for seed in range(20):
        sim = simulate(replace(NOISELESS, max_offset=1e-3), seed)
        if sim.truth.beta[0] < 0:
            break
    clock_T, clock_t = sim.truth.T, sim.truth.t
    inside = sim.uwb.t_rx_true <= clock_t[-1]
    expected = np.interp(sim.uwb.t_rx_true[inside], clock_t, clock_T)
    assert sim.uwb.t_rx_true.min() >= clock_t[0]
    np.testing.assert_allclose(sim.uwb.T_rx[inside], expected, atol=1e-15)


def test_uwb_receptions_satisfy_light_time():
    sim = simulate(NOISELESS, seed=1)
    uwb = sim.uwb
    traj = CircleTrajectory(NOISELESS.speed, NOISELESS.radius, NOISELESS.height)
    rng = np.linalg.norm(traj.position(uwb.t_rx_true) - sim.anchors[uwb.anchor], axis=-1)
    np.testing.assert_allclose(C * (uwb.t_rx_true - uwb.t_tx), rng, atol=1e-6)


def _dead_reckon(sim, coupled):
    """Integrate noiseless IMU from the true initial state; return final position error."""
    x = truth_state(sim.truth, 0)
    for k in range(len(sim.imu.T)):
        x, _ = propagate(x, sim.imu.gyro[k], sim.imu.accel[k], sim.imu.dT, coupled)
    final = truth_state(sim.truth, len(sim.imu.T) - 1)
    # x is one step past the last truth sample; step the truth forward analytically.
    traj = CircleTrajectory(sim.config.speed, sim.config.radius, sim.config.height)
    t_end = sim.truth.t[-1] + sim.imu.dT * (1 - final.gamma / C)
    return np.linalg.norm(x.p - traj.position(t_end))


def test_coupled_dead_reckoning_tracks_truth_with_large_skew():
    sim = simulate(NOISELESS, seed=2)
    assert _dead_reckon(sim, coupled=True) < 1e-3


def test_decoupled_dead_reckoning_drifts_with_large_skew():
    sim = simulate(NOISELESS, seed=2)
    assert _dead_reckon(sim, coupled=False) > 10 * _dead_reckon(sim, coupled=True)


def test_decoupled_matches_coupled_without_skew():
    sim = simulate(replace(NOISELESS, skew=0.0), seed=3)
    assert abs(_dead_reckon(sim, coupled=False) - _dead_reckon(sim, coupled=True)) < 1e-9
