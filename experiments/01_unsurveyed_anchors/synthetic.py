"""Experiment 01 on synthetic flights: unsurveyed anchors.

Each filter starts from anchor positions that are off by SIGMA_ANCHOR per axis
(a tape-measure guess) and estimates them with the drone. Same flights,
clocks and sensors as experiment 00 (circles at 0 to 5 m/s, eight anchors on a
6 x 6 x 2.6 m box), but longer, since an EKF-SLAM's overconfidence builds up
over time. Filters:

    EKF            ekf_slam.EkfSlam, anchors as additive states
    FEJ-EKF        ekf_fej.EkfFej, the same with first-estimate Jacobians
    EqF            eqf_slam.EqfSlam, anchors inside the symmetry group
    EKF surveyed   experiment 00's EKF given the true anchor positions (reference)
    EqF surveyed   experiment 00's EqF given the true anchor positions (reference)

    sbatch slurm/experiment.sbatch experiments/01_unsurveyed_anchors/synthetic.py --out results/01_unsurveyed_anchors/synthetic

Faster flights. In the default scene the circle radius is fixed at 2 m, so the
turn acceleration v^2 / r grows with speed (12.5 m/s^2 at 5 m/s, 50 at 10).
--turn-accel a instead scales the whole scene (anchor box, radius, height) with
v^2 so every circle has acceleration a, as in experiment 00's high-speed sweep;
the anchor survey error stays in metres unless --scale-survey also multiplies
it by the scene scale (same geometry relative to the scene at every speed).
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gal3uwb.eval import runner  # noqa: E402
from gal3uwb.filters.common import FilterNoise  # noqa: E402
from gal3uwb.filters.ekf import Ekf  # noqa: E402
from gal3uwb.filters.eqf import Eqf  # noqa: E402
from gal3uwb.sim.scenario import SCENARIOS, SimConfig, simulate  # noqa: E402

from ekf_fej import EkfFej  # noqa: E402
from ekf_slam import EkfSlam  # noqa: E402
from eqf_slam import EqfSlam  # noqa: E402
from model import SIGMA_ANCHOR, SlamEstimate, initial_estimate, prior_covariance, run_filter, summarize  # noqa: E402

SPEEDS = [0.0, 0.5, 2.0, 5.0]  # [m/s]
SLAM = {"EKF": EkfSlam, "FEJ-EKF": EkfFej, "EqF": EqfSlam}
SURVEYED = {"EKF surveyed": Ekf, "EqF surveyed": Eqf}
BASE_RADIUS = 2.0  # circle radius at scene scale 1 [m]
WARMUP = 5.0  # [s]
RECORD_EVERY = 20  # IMU steps between records (10 Hz)
SERIES = ("nees_native", "nees_yaw", "nees_anchor_pos", "anchor_rmse")
METRICS = ("pos_rmse", "yaw_rmse_deg", "offset_rmse_m", "anchor_rmse", "anchor_h_rmse", "nees_native", "nees_yaw",
           "nees_full", "nees_rot", "nees_pos", "nees_vel", "nees_clock", "nees_anchor_pos", "nees_all", "failed")
FAILED = {m: np.nan for m in METRICS} | {"failed": 1.0}  # a filter that blew up (singular S or non-finite state)


def run_task(task):
    scenario, i_speed, speed, scale, seed, duration, sigma_anchor, nu = task
    clock = SCENARIOS[scenario]
    sim = simulate(SimConfig(speed=speed, clock=clock, duration=duration, radius=BASE_RADIUS, scale=scale,
                             gyro_bias_sigma=0.002), seed)
    n, N = sim.truth.anchor_offset.shape[1], len(sim.uwb.anchors)
    sigma = {**runner.INITIAL_SIGMA, "pos": 0.1, "gyro_bias": 0.002, "accel_bias": 0.05,
             "rho": max(runner.INITIAL_SIGMA["rho"], clock.skew_sigma)}
    P0 = prior_covariance(sigma, n, N, sigma_anchor)
    truth0 = SlamEstimate(runner.truth_estimate(sim.truth, 0), sim.uwb.anchors)
    x0 = initial_estimate(truth0, P0, np.random.default_rng(seed + 1_000_000))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, 0.1, nu_sigma0=nu)
    out = {}
    for name, cls in SLAM.items():
        try:
            run = run_filter(sim, cls, x0, P0, noise, RECORD_EVERY)
        except (np.linalg.LinAlgError, ValueError):
            run = None
        if run is None or not np.all(np.isfinite(run.err)):
            out[name] = {"metrics": dict(FAILED)}
            continue
        dof = run.err.shape[1]
        anchors = run.err[:, -3 * N:].reshape(len(run.T), N, 3)
        out[name] = {"metrics": summarize(run, WARMUP, n, N) | {"failed": 0.0}, "T": run.T,
                     "series": {"nees_native": run.nees_native / dof, "nees_yaw": run.nees_yaw,
                                "nees_anchor_pos": run.nees["anchor_pos"] / (3 * N),
                                "anchor_rmse": np.sqrt(np.mean(np.sum(anchors**2, axis=2), axis=1))}}
    k = len(P0) - 3 * N
    for name, cls in SURVEYED.items():
        try:
            run = runner.run_filter(sim, cls, x0.nav, P0[:k, :k], noise, record_every=RECORD_EVERY)
            m = runner.summarize_run(run, WARMUP)
        except (np.linalg.LinAlgError, ValueError):
            m = None
        if m is None or not np.isfinite(m["pos_rmse"]):
            out[name] = {"metrics": dict(FAILED)}
            continue
        out[name] = {"metrics": {key: m[key] for key in ("pos_rmse", "yaw_rmse_deg", "offset_rmse_m", "nees_native",
                                                         "nees_yaw", "nees_full", "nees_rot", "nees_pos", "nees_vel",
                                                         "nees_clock")} | {"failed": 0.0}}
    return scenario, i_speed, seed, out


def sweep(scenarios, speeds, scales, n_runs, duration, workers, sigma_anchors, nu):
    tasks = [(s, i, v, scales[i], seed, duration, sigma_anchors[i], nu) for s in scenarios
             for i, v in enumerate(speeds) for seed in range(n_runs)]
    shape = (len(scenarios), len(speeds), n_runs)
    res, t0 = {}, time.time()
    with Pool(workers) as pool:
        for done, (scenario, i, seed, out) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            si = scenarios.index(scenario)
            for name, r in out.items():
                for m, v in r["metrics"].items():
                    res.setdefault(f"{name}|{m}", np.full(shape, np.nan))[si, i, seed] = v
                for key, series in r.get("series", {}).items():
                    arr = res.setdefault(f"{name}|series_{key}", np.full(shape + (len(series),), np.nan))
                    arr[si, i, seed] = series
                if "T" in r:
                    res["T"] = r["T"]
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  {done}/{len(tasks)} runs ({time.time() - t0:.0f} s)", flush=True)
    return res


def filters_in(res):
    return list(dict.fromkeys(k.split("|")[0] for k in res if "|" in k))


def aggregate(v, m):
    """RMSEs combine as root-mean-square over runs; NEES as means."""
    return np.sqrt(np.nanmean(v**2, axis=-1)) if "rmse" in m else np.nanmean(v, axis=-1)


def write_csv(res, scenarios, speeds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "speed_mps", "filter"] + list(METRICS))
        for si, scenario in enumerate(scenarios):
            for i, speed in enumerate(speeds):
                for name in filters_in(res):
                    row = []
                    for m in METRICS:
                        key = f"{name}|{m}"
                        row.append(f"{aggregate(res[key][si, i], m):.6g}" if key in res else "")
                    w.writerow([scenario, speed, name] + row)


def print_table(res, scenarios, speeds):
    print(f"\n{'scenario':>14} {'v':>4} {'filter':>13} | {'pos cm':>6} {'yaw deg':>7} {'anch cm':>7} | "
          f"{'native':>6} {'yaw':>5} {'nav':>5} {'anchors':>7} {'all':>5} {'failed':>6}")
    for si, scenario in enumerate(scenarios):
        for i, speed in enumerate(speeds):
            for name in filters_in(res):
                def g(m, scale=1.0, fmt="6.2f"):
                    key = f"{name}|{m}"
                    return format(scale * aggregate(res[key][si, i], m), fmt) if key in res else "-"
                print(f"{scenario:>14} {speed:4.1f} {name:>13} | {g('pos_rmse', 100):>6} {g('yaw_rmse_deg', 1, '7.3f'):>7} "
                      f"{g('anchor_rmse', 100, '7.2f'):>7} | {g('nees_native'):>6} {g('nees_yaw', 1, '5.2f'):>5} "
                      f"{g('nees_full', 1, '5.2f'):>5} {g('nees_anchor_pos', 1, '7.2f'):>7} {g('nees_all', 1, '5.2f'):>5} "
                      f"{g('failed', 1, '6.2f'):>6}")


COLORS = {"EKF": "#eb6834", "FEJ-EKF": "#1baf7a", "EqF": "#2a78d6"}
INK, INK_2, SURFACE, GRID = "#16191e", "#4a505b", "#ffffff", "#e1e5ea"


def plot_series(res, scenarios, speeds, path):
    """Mean NEES over time across runs: full state, heading and anchor positions."""
    T = res["T"]
    rows = [("nees_native", "NEES / dof"), ("nees_yaw", "heading NEES"), ("nees_anchor_pos", "anchor NEES / dof"),
            ("anchor_rmse", "anchor error [cm]")]
    fig, axes = plt.subplots(len(rows), len(scenarios) * len(speeds), figsize=(2.6 * len(scenarios) * len(speeds),
                             2.2 * len(rows)), sharex=True, squeeze=False, facecolor=SURFACE)
    for c, (si, i) in enumerate((si, i) for si in range(len(scenarios)) for i in range(len(speeds))):
        for r, (key, label) in enumerate(rows):
            ax = axes[r, c]
            ax.grid(True, color=GRID)
            for name, color in COLORS.items():
                if f"{name}|series_{key}" not in res:
                    continue
                arr = res[f"{name}|series_{key}"][si, i]
                y = np.sqrt(np.nanmean(arr**2, axis=0)) * 100 if key == "anchor_rmse" else np.nanmean(arr, axis=0)
                ax.plot(T, y, color=color, lw=1.5, label=name)
            if key != "anchor_rmse":
                ax.axhline(1.0, color=INK_2, lw=1, ls=":")
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{scenarios[si]}, {speeds[i]:g} m/s", fontsize=8, loc="left")
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel("time [s]", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--duration", type=float, default=60.0, help="flight duration [s]")
    parser.add_argument("--speeds", nargs="+", type=float, default=SPEEDS)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--sigma-anchor", type=float, default=SIGMA_ANCHOR, help="prior std of each anchor axis [m]")
    parser.add_argument("--turn-accel", type=float, default=None,
                        help="scale the scene so each circle has this centripetal acceleration [m/s^2]")
    parser.add_argument("--scale-survey", action="store_true",
                        help="with --turn-accel, multiply the anchor survey error by the scene scale")
    parser.add_argument("--nu", type=float, default=3e-3, help="prior std of the EqF's virtual bias b_nu [m/s]")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/01_unsurveyed_anchors/synthetic"))
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        with np.load(args.out / "runs.npz", allow_pickle=False) as data:
            res = {k: data[k] for k in data.files}
    else:
        if args.turn_accel is None:
            if args.scale_survey:
                parser.error("--scale-survey needs --turn-accel")
            scales = [1.0] * len(args.speeds)
        else:
            if min(args.speeds) <= 0:
                parser.error("--turn-accel needs positive speeds")
            scales = [v**2 / args.turn_accel / BASE_RADIUS for v in args.speeds]
        sigma_anchors = [args.sigma_anchor * (k if args.scale_survey else 1.0) for k in scales]
        for v, k, sa in zip(args.speeds, scales, sigma_anchors):
            print(f"  {v:g} m/s: radius {BASE_RADIUS * k:.1f} m, turn {v**2 / (BASE_RADIUS * k):.1f} m/s^2, "
                  f"scene x{k:.2f}, anchor prior {sa:.2f} m", flush=True)
        n_tasks = len(args.scenarios) * len(args.speeds) * args.runs
        print(f"{n_tasks} flights of {args.duration:g} s, b_nu prior {args.nu:g}, {args.workers} workers", flush=True)
        t0 = time.time()
        res = sweep(args.scenarios, args.speeds, scales, args.runs, args.duration, args.workers, sigma_anchors, args.nu)
        print(f"done in {time.time() - t0:.0f} s")
        res["scenarios"], res["speeds"] = np.array(args.scenarios), np.array(args.speeds)
        res["scales"], res["sigma_anchors"] = np.array(scales), np.array(sigma_anchors)
        np.savez(args.out / "runs.npz", **res)
    scenarios, speeds = [str(s) for s in res.pop("scenarios")], [float(v) for v in res.pop("speeds")]
    res.pop("scales", None), res.pop("sigma_anchors", None)
    write_csv(res, scenarios, speeds, args.out / "summary.csv")
    plot_series(res, scenarios, speeds, args.out / "nees_over_time.png")
    print_table(res, scenarios, speeds)
    print(f"\nwrote {args.out}/summary.csv, runs.npz, nees_over_time.png")


if __name__ == "__main__":
    main()
