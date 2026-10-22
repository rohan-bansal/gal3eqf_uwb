import numpy as np
import pytest

from gal3uwb.eval.runner import initial_estimate, prior_covariance, run_filter, summarize_run, truth_estimate
from gal3uwb.filters.common import FilterNoise
from gal3uwb.filters.ekf import Ekf, range_model
from gal3uwb.filters.eqf import Eqf
from gal3uwb.params import C
from gal3uwb.sim.scenario import SCENARIOS, SimConfig, simulate


def test_uwb_measurements_match_truth_at_transmission():
    sim = simulate(SimConfig(speed=2.0, clock=SCENARIOS["gaussian_skew"], duration=3.0, sigma_range=0.0), seed=3)
    k = 100  # an IMU instant; build a transmission there and compare with the model
    T_k = sim.truth.T[k]
    j = np.argmin(np.abs(sim.uwb.T_tx - T_k))
    assert abs(sim.uwb.T_tx[j] - T_k) < 0.05
    i = np.searchsorted(sim.truth.T, sim.uwb.T_tx[j])
    # Linear interpolation of the truth between IMU instants is good to ~1e-5 m here.
    w = (sim.uwb.T_tx[j] - sim.truth.T[i - 1]) / (sim.truth.T[i] - sim.truth.T[i - 1])
    x0, x1 = truth_estimate(sim.truth, i - 1), truth_estimate(sim.truth, i)
    x = x0.retract(w * x0.error_to(x1))
    y, _ = range_model(x, sim.uwb.anchors, sim.lever_arm)
    np.testing.assert_allclose(sim.uwb.y[j], y, atol=1e-4)


def test_clock_truth_follows_skew():
    sim = simulate(SimConfig(clock=SCENARIOS["gaussian_skew"], duration=2.0), seed=4)
    d_delta = np.diff(sim.truth.delta)
    expected = -sim.truth.rho[:-1] * np.diff(sim.truth.T)
    assert np.abs(d_delta - expected).max() < 2e-10  # white offset noise, sigma 2e-11 per step
    ideal = simulate(SimConfig(clock=SCENARIOS["ideal_clock"], duration=2.0), seed=4)
    assert np.ptp(ideal.truth.delta) == 0.0 and np.all(ideal.truth.rho == 0.0)


@pytest.mark.parametrize("cls", [Ekf, Eqf])
def test_filters_track_a_short_flight(cls):
    config = SimConfig(speed=2.0, clock=SCENARIOS["gaussian_skew"], duration=6.0)
    sim = simulate(config, seed=5)
    P0 = prior_covariance()
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(6))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, config.sigma_range)
    run = run_filter(sim, cls, x0, P0, noise, record_every=20)
    m = summarize_run(run, warmup=2.0)
    assert m["pos_rmse"] < 0.1
    assert m["offset_rmse_m"] < 0.1
    assert 0.3 < m["nees_full"] < 3.0
    assert 0.3 < m["nees_native"] < 3.0


def _interp_truth(sim, T):
    """Truth at an arbitrary tag time, linearly interpolated between IMU instants."""
    i = np.searchsorted(sim.truth.T, T)
    w = (T - sim.truth.T[i - 1]) / (sim.truth.T[i] - sim.truth.T[i - 1])
    x0, x1 = truth_estimate(sim.truth, i - 1), truth_estimate(sim.truth, i)
    return x0.retract(w * x0.error_to(x1))


def test_unsynced_measurements_match_truth():
    config = SimConfig(speed=2.0, clock=SCENARIOS["unsynced"], duration=3.0, sigma_range=0.0)
    sim = simulate(config, seed=7)
    assert sim.truth.anchor_offset.shape[1] == len(sim.uwb.anchors) - 1
    j = 30
    y, _ = range_model(_interp_truth(sim, sim.uwb.T_tx[j]), sim.uwb.anchors, sim.lever_arm)
    np.testing.assert_allclose(sim.uwb.y[j], y, atol=1e-3)
    from gal3uwb.filters.ekf import anchor_columns, anchor_offsets, sync_model

    for m in (5, 77):
        x = _interp_truth(sim, sim.sync.T[m])
        n = len(sim.uwb.anchors)
        h, _, rx = sync_model(x.dim, anchor_offsets(x, n), anchor_columns(x, n), sim.sync.tx[m], sim.uwb.anchors)
        np.testing.assert_allclose(sim.sync.y[m][rx], h, atol=1e-3)
        assert np.isnan(sim.sync.y[m][sim.sync.tx[m]])


def test_unsynced_clock_truth_follows_skews():
    sim = simulate(SimConfig(clock=SCENARIOS["unsynced"], duration=2.0), seed=8)
    dT = np.diff(sim.truth.T)[:, None]
    np.testing.assert_allclose(np.diff(sim.truth.anchor_offset, axis=0), sim.truth.anchor_skew[:-1] * dT, atol=3e-10)
    np.testing.assert_allclose(np.diff(sim.truth.delta), -sim.truth.rho[:-1] * dT[:, 0], atol=3e-10)


@pytest.mark.parametrize("cls", [Ekf, Eqf])
def test_filters_track_a_short_unsynced_flight(cls):
    config = SimConfig(speed=2.0, clock=SCENARIOS["unsynced"], duration=6.0)
    sim = simulate(config, seed=9)
    n = sim.truth.anchor_offset.shape[1]
    P0 = prior_covariance(n=n)
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(10))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, config.sigma_range)
    m = summarize_run(run_filter(sim, cls, x0, P0, noise, record_every=20), warmup=2.0)
    assert m["pos_rmse"] < 0.1
    assert m["anchor_offset_rmse_m"] < 0.1
    assert 0.3 < m["nees_full"] < 3.0
    assert 0.3 < m["nees_native"] < 3.0
