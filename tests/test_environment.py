"""Smoke tests for the GTSAM install and the Gal3 conventions we rely on."""

import gtsam
import numpy as np

I3 = gtsam.Rot3()
Z3 = np.zeros(3)


def test_gal3_constructor_order_is_R_p_v_t():
    g = gtsam.Gal3(I3, np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), 7.0)
    np.testing.assert_allclose(g.position(), [1, 2, 3])
    np.testing.assert_allclose(g.velocity(), [4, 5, 6])
    assert g.time() == 7.0


def test_gal3_matrix_layout():
    # [[R, v, p], [0, 1, t], [0, 0, 1]]
    g = gtsam.Gal3(I3, np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), 7.0)
    M = g.matrix()
    np.testing.assert_allclose(M[:3, 3], [4, 5, 6])
    np.testing.assert_allclose(M[:3, 4], [1, 2, 3])
    assert M[3, 4] == 7.0


def test_gal3_tangent_order_is_omega_v_p_t():
    log = gtsam.Gal3.Logmap
    np.testing.assert_allclose(log(gtsam.Gal3(I3, Z3, np.array([1.0, 2, 3]), 0.0))[3:6], [1, 2, 3])
    np.testing.assert_allclose(log(gtsam.Gal3(I3, np.array([1.0, 2, 3]), Z3, 0.0))[6:9], [1, 2, 3])
    assert log(gtsam.Gal3(I3, Z3, Z3, 5.0))[9] == 5.0


def test_gal3_exp_log_roundtrip():
    xi = np.linspace(-0.5, 0.5, 10)
    np.testing.assert_allclose(gtsam.Gal3.Logmap(gtsam.Gal3.Expmap(xi)), xi, atol=1e-12)


def test_scenario_runner_produces_imu():
    scenario = gtsam.ConstantTwistScenario(np.array([0, 0, 0.1]), np.array([1.0, 0, 0]))
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    runner = gtsam.ScenarioRunner(scenario, params, 0.005, gtsam.imuBias.ConstantBias())
    assert runner.actualAngularVelocity(1.0).shape == (3,)
    assert runner.actualSpecificForce(1.0).shape == (3,)
