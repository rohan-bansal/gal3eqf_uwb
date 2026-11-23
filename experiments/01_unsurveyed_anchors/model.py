"""State, prior and run loop shared by the two filters of experiment 01.

The anchor positions are not surveyed: each filter starts from rough positions
(std SIGMA_ANCHOR per axis, a tape-measure guess) and estimates them along with
the drone, its biases and the clocks.

Both filters are scored in the same physical error coordinates, size 17 + 2n + 3N:
    [theta (body frame), dv, dp, d gyro bias, d accel bias, d delta, d rho   (common.py, 17)
     d Delta_1..n, d kappa_1..n                                               (anchor clocks, step 3)
     d a_0, ..., d a_{N-1}]                                                   (anchor positions)
"""

from dataclasses import dataclass, replace

import numpy as np

from gal3uwb.eval.runner import blocks as nav_blocks
from gal3uwb.eval.runner import prior_covariance as nav_prior
from gal3uwb.eval.runner import truth_estimate
from gal3uwb.filters.common import DELTA, DIM, POS, ROT, Estimate, nees
from gal3uwb.params import C

SIGMA_ANCHOR = 0.3  # prior std of each anchor coordinate [m]


@dataclass(frozen=True)
class SlamEstimate:
    nav: Estimate  # drone, biases, tag clock and anchor clocks (common.Estimate)
    anchors: np.ndarray  # (N, 3) anchor positions [m]

    @property
    def dim(self):
        return self.nav.dim + 3 * len(self.anchors)

    def retract(self, e):
        k = self.nav.dim
        return SlamEstimate(self.nav.retract(e[:k]), self.anchors + e[k:].reshape(-1, 3))

    def error_to(self, truth):
        return np.r_[self.nav.error_to(truth.nav), (truth.anchors - self.anchors).ravel()]


def prior_covariance(sigma, n_clocks, n_anchors, sigma_anchor=SIGMA_ANCHOR):
    P_nav = nav_prior(sigma, n_clocks)
    k = len(P_nav)
    P = np.zeros((k + 3 * n_anchors, k + 3 * n_anchors))
    P[:k, :k] = P_nav
    P[k:, k:] = sigma_anchor**2 * np.eye(3 * n_anchors)
    return P


def initial_estimate(truth: SlamEstimate, P0, rng) -> SlamEstimate:
    """Drawn from the prior around the truth; biases and skews start at zero, which
    matches the prior because their truth is drawn from it."""
    e = rng.multivariate_normal(np.zeros(len(P0)), P0)
    x = truth.retract(-e)
    nav = replace(x.nav, gyro_bias=np.zeros(3), accel_bias=np.zeros(3), rho=0.0,
                  anchor_skew=np.zeros(x.nav.n_anchor_clocks))
    return SlamEstimate(nav, x.anchors)


def blocks(n_clocks, n_anchors):
    """NEES blocks in the physical coordinates: the navigation blocks of runner.py
    plus "anchor_pos" (all anchor positions) and "all" (everything)."""
    k = DIM + 2 * n_clocks
    out = nav_blocks(n_clocks)
    out["anchor_pos"] = np.arange(k, k + 3 * n_anchors)
    out["all"] = np.arange(k + 3 * n_anchors)
    return out


@dataclass
class Run:
    T: np.ndarray  # (M,) record times [s]
    err: np.ndarray  # (M, dim) physical error
    nees: dict  # block -> (M,) NEES in the physical coordinates
    nees_native: np.ndarray  # (M,) full NEES in the filter's own error coordinates
    yaw_err: np.ndarray  # (M,) world-frame yaw error [rad]
    nees_yaw: np.ndarray


def run_filter(sim, cls, x0: SlamEstimate, P0, noise, record_every=20) -> Run:
    """Run a filter over a flight. Unlike runner.run_filter, the filter is never told
    the anchor positions: it uses its own estimates."""
    imu, uwb, sync, truth = sim.imu, sim.uwb, sim.sync, sim.truth
    f = cls(x0, P0, noise)
    n_clocks, n_anchors = x0.nav.n_anchor_clocks, len(x0.anchors)
    idx = blocks(n_clocks, n_anchors)
    rec = {"T": [], "err": [], "native": [], "yaw": [], "nees_yaw": [], **{b: [] for b in idx}}

    def record(k):
        x_true = SlamEstimate(truth_estimate(truth, k), uwb.anchors)
        x_hat = f.estimate()
        e, P = x_hat.error_to(x_true), f.physical_covariance()
        rec["T"].append(imu.T[k])
        rec["err"].append(e)
        for name, i in idx.items():
            rec[name].append(nees(e[i], P[np.ix_(i, i)]))
        eps, P_native = f.native_error(x_true)
        rec["native"].append(nees(eps, P_native))
        theta_w = x_hat.nav.R @ e[ROT]
        P_w = x_hat.nav.R @ P[ROT, ROT] @ x_hat.nav.R.T
        rec["yaw"].append(theta_w[2])
        rec["nees_yaw"].append(theta_w[2] ** 2 / P_w[2, 2])

    events = [(T, 0, j) for j, T in enumerate(uwb.T_tx)]
    if sync is not None:
        events += [(T, 1, m) for m, T in enumerate(sync.T)]
    events.sort()
    j = 0
    record(0)
    for k in range(len(imu.T) - 1):
        T, T_end = imu.T[k], imu.T[k + 1]
        g, a = imu.gyro[k], imu.accel[k]
        while j < len(events) and events[j][0] < T_end:
            T_event, kind, i = events[j]
            if T_event > T:
                f.propagate(g, a, T_event - T)
                T = T_event
            if kind == 0:
                f.update(uwb.y[i], sim.lever_arm)
            else:
                f.update_sync(sync.y[i], sync.tx[i])
            j += 1
        if T_end > T:
            f.propagate(g, a, T_end - T)
        if (k + 1) % record_every == 0:
            record(k + 1)

    return Run(np.array(rec["T"]), np.array(rec["err"]), {b: np.array(rec[b]) for b in idx},
               np.array(rec["native"]), np.array(rec["yaw"]), np.array(rec["nees_yaw"]))


def summarize(run: Run, warmup, n_clocks, n_anchors):
    """Scalar metrics after the warm-up; NEES are per degree of freedom (1 is consistent)."""
    keep = run.T >= warmup
    e = run.err[keep]
    k = DIM + 2 * n_clocks

    def rms(x):
        x = x.reshape(len(x), -1)
        return float(np.sqrt(np.mean(np.sum(x**2, axis=1))))

    anchors = e[:, k:].reshape(len(e), n_anchors, 3)
    out = {
        "pos_rmse": rms(e[:, POS]),
        "pos_h_rmse": rms(e[:, 6:8]),
        "pos_v_rmse": rms(e[:, 8]),
        "yaw_rmse_deg": np.degrees(rms(run.yaw_err[keep])),
        "offset_rmse_m": C * rms(e[:, DELTA]),
        "anchor_rmse": float(np.sqrt(np.mean(np.sum(anchors**2, axis=2)))),  # per anchor, 3D
        "anchor_h_rmse": float(np.sqrt(np.mean(np.sum(anchors[..., :2] ** 2, axis=2)))),
        "nees_native": float(run.nees_native[keep].mean()) / (k + 3 * n_anchors),
        "nees_yaw": float(run.nees_yaw[keep].mean()),
    }
    for name, i in blocks(n_clocks, n_anchors).items():
        out[f"nees_{name}"] = float(run.nees[name][keep].mean()) / len(i)
    return out
