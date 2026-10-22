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
