"""EqF on Gal(3) vs EKF on real MILUV flights, for the three clock scenarios.

The drone path (mocap) and the PX4 IMU are real; the tag clock, anchor clocks
(for step 3) and all timestamps are simulated at MILUV's own anchor and tag
positions. Each (flight, scenario, seed) draws a new clock, timestamp noise and
initial estimate, and runs every filter on the same data.

The filters' IMU white noise comes from the airborne IMU itself (vibration
dominates the static Allan values); the bias random walks are 10x the static
calibration. The bias "truth" is a sliding-window estimate against mocap, so
consistency is judged on position, velocity and clock ("pvc"); the mocap
attitude also carries a few degrees of IMU-to-body mounting tilt.

    sbatch slurm/experiment.sbatch experiments/00_eqf_vs_ekf/miluv.py --scenarios unsynced --out results/miluv_step3

Steps 1-2 in results/miluv_steps12 were run with --nu-loose 1e-2.
"""

import argparse
import csv
import os
import pickle
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from gal3uwb.data.miluv import Flight, imu_calibration, simulate_on_flight
from gal3uwb.eval.runner import INITIAL_SIGMA, initial_estimate, prior_covariance, run_filter, summarize_run, truth_estimate
from gal3uwb.filters.common import FilterNoise
from gal3uwb.filters.ekf import Ekf
from gal3uwb.filters.eqf import Eqf
from gal3uwb.params import ImuNoise
from gal3uwb.sim.scenario import SCENARIOS

FLIGHTS = [
    "default_1_random3_0", "default_1_random3_1", "default_1_random3_2", "default_1_circular2D_0",
    "default_1_circular3D_0", "default_1_remoteControlLowPace_0", "default_1_remoteControlLowPace_0_v2",
    "default_1_remoteControlHighPace_0", "default_1_remoteControlHighPace_1", "default_1_remoteControlHighPace_2",
]
FILTERS = {"EKF": Ekf, "EqF": Eqf, "EqF-loose": Eqf}
NU_TIGHT = 1e-4
SCENARIO_LABELS = {"ideal_clock": "step 1: ideal clock", "gaussian_skew": "step 2: Gaussian skew",
                   "unsynced": "step 3: unsynced anchors"}
UWB_RATE = 20.0  # [Hz]
SIGMA_RANGE = 0.1  # [m]
WARMUP = 10.0  # [s]
RECORD_EVERY = 25  # IMU steps between records (10 Hz at 250 Hz)
# The gyro bias is calibrated at rest before takeoff; MILUV's PX4 shows ~0.5 m/s^2 of
# apparent accel bias (bias plus mounting tilt), which cannot be calibrated at rest.
PRIOR = {**INITIAL_SIGMA, "pos": 0.1, "gyro_bias": 0.005, "accel_bias": 0.5}
METRICS = ("pos_rmse", "pos_h_rmse", "pos_v_rmse", "vel_rmse", "rot_rmse_deg", "yaw_rmse_deg", "offset_rmse_m",
           "anchor_offset_rmse_m", "nees_pvc", "nees_native", "nees_full", "nees_rot", "nees_yaw", "nees_vel",
           "nees_pos", "nees_clock", "nees_anchors")


def filter_labels(nu_loose):
    return {"EKF": "EKF", "EqF": f"EqF, b_nu prior {NU_TIGHT:g}", "EqF-loose": f"EqF, b_nu prior {nu_loose:g}"}

_FLIGHTS = {}


def prepare(name):
    """Load a flight, fit the mocap splines and estimate biases and IMU noise once."""
    flight = Flight.load(name)
    flight.traj.position(flight.t_imu[:2])
    flight.traj.rotation(flight.t_imu[:2])
    biases = flight.bias_estimate()
    gyro_nd, accel_nd = flight.vibration_noise()
    static = imu_calibration(flight.robot)
    noise = ImuNoise(gyro_nd, accel_nd, 10 * static.gyro_bias_rw, 10 * static.accel_bias_rw)
    return name, {"flight": flight, "biases": biases, "imu_noise": noise, "speed": flight.speed_stats(),
                  "gyro_bias0": flight.static_gyro_bias()}


def _load_cache(path):
    global _FLIGHTS
    with open(path, "rb") as f:
        _FLIGHTS = pickle.load(f)


def run_task(task):
    name, scenario, seed, nu_loose = task
    entry = _FLIGHTS[name]
    clock = SCENARIOS[scenario]
    sim = simulate_on_flight(entry["flight"], clock, UWB_RATE, SIGMA_RANGE, seed, biases=entry["biases"])
    sim = replace(sim, imu_noise=entry["imu_noise"])
    P0 = prior_covariance(PRIOR, sim.truth.anchor_offset.shape[1])
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(seed + 1_000_000))
    x0 = replace(x0, gyro_bias=entry["gyro_bias0"])
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, SIGMA_RANGE)
    nu = {"EKF": NU_TIGHT, "EqF": NU_TIGHT, "EqF-loose": nu_loose}
    out = {}
    for fname, cls in FILTERS.items():
        run = run_filter(sim, cls, x0, P0, replace(noise, nu_sigma0=nu[fname]), record_every=RECORD_EVERY)
        out[fname] = {"metrics": summarize_run(run, WARMUP), "T": run.T - run.T[0],
                      "nees_pvc": run.nees["pvc"] / 8, "nees_yaw": run.nees_yaw}
    return name, scenario, seed, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--flights", nargs="+", default=FLIGHTS)
    parser.add_argument("--nu-loose", type=float, default=3e-3, help="b_nu prior of the second EqF [m/s]")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/miluv"))
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "flights.pkl"

    if args.replot:
        with open(args.out / "runs.pkl", "rb") as f:
            saved = pickle.load(f)
    else:
        t0 = time.time()
        with Pool(min(args.workers, len(args.flights))) as pool:
            flights = dict(pool.map(prepare, args.flights))
        with open(cache, "wb") as f:
            pickle.dump(flights, f)
        print(f"prepared {len(flights)} flights in {time.time() - t0:.0f} s", flush=True)
        for name, e in flights.items():
            n = e["imu_noise"]
            print(f"  {name:38s} speed mean/p95/max {e['speed'][0]:.2f}/{e['speed'][1]:.2f}/{e['speed'][2]:.2f} m/s, "
                  f"airborne IMU noise gyro {n.gyro_nd:.1e} accel {n.accel_nd:.1e}", flush=True)

        tasks = [(n, s, seed, args.nu_loose) for n in args.flights for s in args.scenarios for seed in range(args.seeds)]
        print(f"{len(tasks)} runs x {len(FILTERS)} filters on {args.workers} workers", flush=True)
        results = []
        with Pool(args.workers, initializer=_load_cache, initargs=(cache,)) as pool:
            for done, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
                results.append(r)
                if done % max(1, len(tasks) // 10) == 0:
                    print(f"  {done}/{len(tasks)} runs ({time.time() - t0:.0f} s)", flush=True)
        saved = {"results": results, "speed": {n: e["speed"] for n, e in flights.items()},
                 "flights": args.flights, "scenarios": args.scenarios, "seeds": args.seeds, "nu_loose": args.nu_loose}
        with open(args.out / "runs.pkl", "wb") as f:
            pickle.dump(saved, f)

    table = summarize(saved)
    write_csv(table, args.out / "summary.csv")
    print_table(table)
    plot(saved, table, args.out / "summary.png")
    print(f"\nwrote {args.out}/summary.csv, runs.pkl, summary.png")


def summarize(saved):
    table = []
    for name in saved["flights"]:
        for scenario in saved["scenarios"]:
            for fname in FILTERS:
                ms = [r[3][fname]["metrics"] for r in saved["results"] if r[0] == name and r[1] == scenario]
                row = {"flight": name, "scenario": scenario, "filter": fname,
                       "speed_mean": saved["speed"][name][0], "speed_p95": saved["speed"][name][1]}
                for m in METRICS:
                    vals = np.array([x.get(m, np.nan) for x in ms])  # anchor metrics exist only for step 3
                    row[m] = np.sqrt(np.mean(vals**2)) if "rmse" in m else np.mean(vals)
                table.append(row)
    return table


def write_csv(table, path):
    keys = list(table[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for row in table:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def print_table(table):
    print(f"\n{'flight':>36} {'scenario':>13} {'filter':>9} {'v':>4} | {'h cm':>5} {'z cm':>5} {'yaw deg':>7} "
          f"{'offs cm':>7} {'anch cm':>7} | {'pvc':>6} {'native':>6} {'rot':>5} {'yaw':>5} {'vel':>5} {'pos':>5} "
          f"{'clock':>5} {'anch':>5}")
    for r in table:
        print(f"{r['flight']:>36} {r['scenario']:>13} {r['filter']:>9} {r['speed_mean']:4.2f} | "
              f"{100 * r['pos_h_rmse']:5.1f} {100 * r['pos_v_rmse']:5.1f} {r['yaw_rmse_deg']:7.2f} "
              f"{100 * r['offset_rmse_m']:7.2f} {100 * r['anchor_offset_rmse_m']:7.2f} | "
              f"{r['nees_pvc']:6.2f} {r['nees_native']:6.2f} {r['nees_rot']:5.2f} {r['nees_yaw']:5.2f} "
              f"{r['nees_vel']:5.2f} {r['nees_pos']:5.2f} {r['nees_clock']:5.2f} {r['nees_anchors']:5.2f}")


FILTER_COLORS = {"EKF": "#2a78d6", "EqF": "#eb6834", "EqF-loose": "#1baf7a"}
INK, INK_2, SURFACE, GRID, BAND = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#ecebe7"


def plot(saved, table, path):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    })
    # Attitude NEES is left out: the mocap attitude carries a few degrees of IMU mounting tilt.
    panels = [("pos_h_rmse", 100.0, "horizontal position RMSE [cm]", "Horizontal position", False),
              ("pos_v_rmse", 100.0, "vertical position RMSE [cm]", "Vertical position (anchors near flight height)",
               False),
              ("offset_rmse_m", 100.0, "clock offset RMSE, C x delta [cm]", "Clock offset", False),
              ("nees_pvc", 1.0, "NEES / dof, pos + vel + clock (log)", "Consistency", True)]
    scenarios = saved["scenarios"]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.0), facecolor=SURFACE)
    # Small horizontal offsets so the three filters' markers do not hide each other.
    shift = dict(zip(FILTER_COLORS, (-0.012, 0.0, 0.012)))
    for ax, (key, scale, ylabel, title, log) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for k, scenario in enumerate(scenarios):
            for fname, color in FILTER_COLORS.items():
                rows = [r for r in table if r["scenario"] == scenario and r["filter"] == fname]
                x = [r["speed_mean"] + shift[fname] for r in rows]
                y = [scale * r[key] for r in rows]
                ax.plot(x, y, ls="none", marker="o" if k == 0 else "D", ms=6, color=color,
                        mfc=color if k == 0 else SURFACE, mew=1.5)
        if key == "nees_pvc":
            n = saved["seeds"]
            lo, hi = chi2.ppf([0.025, 0.975], n * 8) / (n * 8)
            ax.axhspan(lo, hi, color=BAND, linewidth=0, zorder=0)
            ax.axhline(1.0, color=INK_2, lw=1, ls=":")
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("flight mean speed [m/s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=6, color=c) for c in FILTER_COLORS.values()]
    handles += [plt.Line2D([], [], ls="none", marker=m, ms=6, color=INK_2, mfc=mfc, mew=1.5)
                for m, mfc in (("o", INK_2), ("D", SURFACE))][: len(scenarios)]
    names = filter_labels(saved.get("nu_loose", 1e-2))  # older runs used 1e-2
    labels = [names[n] for n in FILTER_COLORS] + [SCENARIO_LABELS[s] for s in scenarios]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("MILUV flights (real path and IMU) with simulated synchronized-anchor UWB", x=0.01, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
