"""Run a filter over a simulated flight and record errors and NEES."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.filters.common import (BA, BG, DELTA, DIM, POS, RHO, ROT, VEL, Estimate, FilterNoise, anchor_slices,
                                    nees)
from gal3uwb.params import C

# Prior standard deviations; the initial estimate is drawn from this prior.
INITIAL_SIGMA = {
    "rot": np.array([0.02, 0.02, 0.05]),  # [rad], body frame
    "vel": 0.1,  # [m/s]
    "pos": 0.3,  # [m]
    "gyro_bias": 0.01,  # [rad/s]
    "accel_bias": 0.05,  # [m/s^2]
    "delta": 1.0 / C,  # [s], i.e. 1 m: the first packets pin each clock offset down
    "rho": 20e-6,  # skew std of each free-running clock [-]
}


def blocks(n):
    """NEES blocks in the physical error coordinates for n anchor clocks. "clock" is
    the tag clock (delta, rho); "pvc" (position, velocity, tag clock) leaves out
    attitude and biases, whose truth is only approximate on real data."""
    named = {"full": slice(0, DIM + 2 * n), "rot": ROT, "vel": VEL, "pos": POS, "bias": slice(9, 15),
             "clock": slice(15, 17), "pvc": np.r_[3:9, 15:17]}
    if n:
        named["anchors"] = slice(DIM, DIM + 2 * n)
    return {name: np.arange(DIM + 2 * n)[idx] for name, idx in named.items()}


def prior_covariance(sigma=INITIAL_SIGMA, n=0):
    """Prior over the physical error coordinates for n anchor clocks.

    Synchronized anchors (n = 0) are an ideal reference, so rho is the tag's skew.
    Otherwise rho = s_tag - s_0 and kappa_i = s_i - s_0 share anchor 0's skew.
    """
    P = np.zeros((DIM + 2 * n, DIM + 2 * n))
    d = np.zeros(DIM)
    d[ROT] = sigma["rot"] ** 2
    d[VEL] = sigma["vel"] ** 2
    d[POS] = sigma["pos"] ** 2
    d[BG] = sigma["gyro_bias"] ** 2
    d[BA] = sigma["accel_bias"] ** 2
    d[DELTA] = sigma["delta"] ** 2
    d[RHO] = sigma["rho"] ** 2
    P[:DIM, :DIM] = np.diag(d)
    if n:
        off, skew = anchor_slices(n)
        P[off, off] = sigma["delta"] ** 2 * np.eye(n)
        idx = np.r_[RHO, DIM + n:DIM + 2 * n]
        P[np.ix_(idx, idx)] = sigma["rho"] ** 2 * (np.eye(n + 1) + np.ones((n + 1, n + 1)))
    return P


def truth_estimate(truth, k) -> Estimate:
    return Estimate(truth.R[k], truth.v[k], truth.p[k], truth.gyro_bias[k], truth.accel_bias[k], truth.delta[k],
                    truth.rho[k], truth.anchor_offset[k], truth.anchor_skew[k])


def initial_estimate(truth0: Estimate, P0, rng) -> Estimate:
    """Pose, velocity and clock offsets drawn from the prior around the truth. Biases and
    skews start at zero, which matches the prior because their truth is drawn from it."""
    n = truth0.n_anchor_clocks
    e = rng.multivariate_normal(np.zeros(DIM + 2 * n), P0)
    x0 = truth0.retract(-e)
    return Estimate(x0.R, x0.v, x0.p, np.zeros(3), np.zeros(3), x0.delta, 0.0, x0.anchor_offset, np.zeros(n))


@dataclass
class FilterRun:
    T: np.ndarray  # (M,) record times [s]
    err: np.ndarray  # (M, 17) physical error, truth = estimate.retract(err)
    nees: dict  # block -> (M,) NEES in physical coordinates
    nees_native: np.ndarray  # (M,) full NEES in the filter's own error coordinates
    yaw_err: np.ndarray  # (M,) world-frame yaw error [rad]
    nees_yaw: np.ndarray  # (M,)


def run_filter(sim, filter_cls, x0, P0, noise: FilterNoise, record_every=10) -> FilterRun:
    imu, uwb, sync, truth = sim.imu, sim.uwb, sim.sync, sim.truth
    f = filter_cls(x0, P0, noise)
    block_idx = blocks(x0.n_anchor_clocks)
    rec = {"T": [], "err": [], "native": [], "yaw": [], "nees_yaw": [], **{b: [] for b in block_idx}}

    def record(k):
        x_true = truth_estimate(truth, k)
        x_hat = f.estimate()
        e, P = x_hat.error_to(x_true), f.physical_covariance()
        rec["T"].append(imu.T[k])
        rec["err"].append(e)
        for name, idx in block_idx.items():
            rec[name].append(nees(e[idx], P[np.ix_(idx, idx)]))
        eps, P_native = f.native_error(x_true)
        rec["native"].append(nees(eps, P_native))
        # World-frame attitude error R_hat theta; its z component is the yaw error.
        theta_w = x_hat.R @ e[ROT]
        P_w = x_hat.R @ P[ROT, ROT] @ x_hat.R.T
        rec["yaw"].append(theta_w[2])
        rec["nees_yaw"].append(theta_w[2] ** 2 / P_w[2, 2])

    # Tag transmissions (kind 0) and anchor-to-anchor packets (kind 1), in tag-time order.
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
            T_event, kind, idx = events[j]
            if T_event > T:
                f.propagate(g, a, T_event - T)
                T = T_event
            if kind == 0:
                f.update(uwb.y[idx], uwb.anchors, sim.lever_arm)
            else:
                f.update_sync(sync.y[idx], sync.tx[idx], uwb.anchors)
            j += 1
        if T_end > T:
            f.propagate(g, a, T_end - T)
        if (k + 1) % record_every == 0:
            record(k + 1)

    return FilterRun(
        T=np.array(rec["T"]),
        err=np.array(rec["err"]),
        nees={b: np.array(rec[b]) for b in block_idx},
        nees_native=np.array(rec["native"]),
        yaw_err=np.array(rec["yaw"]),
        nees_yaw=np.array(rec["nees_yaw"]),
    )


def summarize_run(run: FilterRun, warmup):
    """Scalar metrics over the part of the run after the warm-up."""
    keep = run.T >= warmup
    e = run.err[keep]
    n = (e.shape[1] - DIM) // 2

    def rms(x):
        x = x.reshape(len(x), -1)
        return float(np.sqrt(np.mean(np.sum(x**2, axis=1))))

    out = {
        "pos_rmse": rms(e[:, POS]),
        "pos_h_rmse": rms(e[:, 6:8]),
        "pos_v_rmse": rms(e[:, 8]),
        "vel_rmse": rms(e[:, VEL]),
        "rot_rmse_deg": np.degrees(rms(e[:, ROT])),
        "yaw_rmse_deg": np.degrees(rms(run.yaw_err[keep])),
        "offset_rmse_m": C * rms(e[:, DELTA]),
        "rho_rmse_ppm": 1e6 * rms(e[:, RHO]),
        # NEES per degree of freedom: 1 is consistent.
        "nees_native": float(run.nees_native[keep].mean()) / (DIM + 2 * n),
        "nees_yaw": float(run.nees_yaw[keep].mean()),
    }
    if n:
        off, _ = anchor_slices(n)
        out["anchor_offset_rmse_m"] = C * float(np.sqrt(np.mean(e[:, off] ** 2)))  # per anchor
    for name, idx in blocks(n).items():
        out[f"nees_{name}"] = float(run.nees[name][keep].mean()) / len(idx)
    return out
