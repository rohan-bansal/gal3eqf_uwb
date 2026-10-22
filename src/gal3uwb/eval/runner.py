"""Run a filter over a simulated flight and score it against ground truth."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.filters.eskf import BETA, DIM, GAMMA, POS, ClockEskf, State
from gal3uwb.params import C

# Initial 1-sigma uncertainty per error-state block.
INITIAL_SIGMA = {
    "rot": np.array([0.02, 0.02, 0.05]),  # [rad]
    "vel": 0.1,  # [m/s]
    "pos": 0.3,  # [m]
    "beta": 1.0,  # [m]
    "gamma": C * 5e-6,  # [m/s], 5 ppm
}


@dataclass(frozen=True)
class FilterRun:
    T: np.ndarray  # tag times of recorded samples
    err: np.ndarray  # (N, DIM) truth relative to estimate
    nees: np.ndarray
    nees_pos: np.ndarray
    nees_clock: np.ndarray
    p_hat: np.ndarray  # (N, 3)


def truth_state(truth, k):
    return State(
        truth.R[k], truth.v[k], truth.p[k], truth.gyro_bias[k], truth.accel_bias[k], truth.beta[k], truth.gamma[k]
    )


def initial_estimate(sim, rng):
    """Truth perturbed by a draw from P0; biases start at zero (their prior)."""
    cfg = sim.config
    sig = np.concatenate(
        [
            INITIAL_SIGMA["rot"],
            np.full(3, INITIAL_SIGMA["vel"]),
            np.full(3, INITIAL_SIGMA["pos"]),
            np.full(3, cfg.gyro_bias_sigma),
            np.full(3, cfg.accel_bias_sigma),
            [INITIAL_SIGMA["beta"], INITIAL_SIGMA["gamma"]],
        ]
    )
    x = truth_state(sim.truth, 0).retract(-sig * rng.standard_normal(DIM))
    x = State(x.R, x.v, x.p, np.zeros(3), np.zeros(3), x.beta, x.gamma)
    return x, np.diag(sig**2)


def _nees(e, P):
    return float(e @ np.linalg.solve(P, e))


def run_filter(sim, coupled, x0, P0, record_every=20):
    cfg, imu, uwb, truth = sim.config, sim.imu, sim.uwb, sim.truth
    filt = ClockEskf(x0, P0, cfg.imu_noise, cfg.clock_noise, coupled)

    order = np.argsort(uwb.T_rx)
    T_rx = uwb.T_rx[order]
    z = C * (T_rx - uwb.t_tx[order])
    anchor_pos = sim.anchors[uwb.anchor[order]]
    j = int(np.searchsorted(T_rx, imu.T[0]))

    rec = {key: [] for key in ("T", "err", "nees", "nees_pos", "nees_clock", "p_hat")}
    clock = [BETA, GAMMA]
    for k in range(len(imu.T)):
        if k % record_every == 0:
            e = filt.x.local(truth_state(truth, k))
            P = filt.P
            rec["T"].append(imu.T[k])
            rec["err"].append(e)
            rec["nees"].append(_nees(e, P))
            rec["nees_pos"].append(_nees(e[POS], P[POS, POS]))
            rec["nees_clock"].append(_nees(e[clock], P[np.ix_(clock, clock)]))
            rec["p_hat"].append(filt.x.p)

        T_now, T_end = imu.T[k], imu.T[k] + imu.dT
        gyro, accel = imu.gyro[k], imu.accel[k]
        while j < len(T_rx) and T_rx[j] < T_end:
            if T_rx[j] > T_now:
                filt.propagate(gyro, accel, T_rx[j] - T_now)
                T_now = T_rx[j]
            filt.update_range(z[j], anchor_pos[j], cfg.sigma_range)
            j += 1
        filt.propagate(gyro, accel, T_end - T_now)

    return FilterRun(**{key: np.array(val) for key, val in rec.items()})
