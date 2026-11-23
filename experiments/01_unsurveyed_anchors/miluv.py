"""Experiment 01 on MILUV flights: unsurveyed anchors.

The real mocap path and PX4 IMU of ten single-drone MILUV flights, with tag
timestamps (and clocks) simulated at MILUV's six anchors, as in experiment 00.
The filters start from anchor positions off by SIGMA_ANCHOR per axis. MILUV's
attitude and bias truth is approximate (IMU mounting tilt), so consistency is
scored on position, velocity and clock ("pvc") and on the anchor positions.

    sbatch slurm/experiment.sbatch experiments/01_unsurveyed_anchors/miluv.py --out results/01_unsurveyed_anchors/miluv
"""

import argparse
import csv
import os
import pickle
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

from gal3uwb.data.miluv import Flight, imu_calibration, simulate_on_flight  # noqa: E402
from gal3uwb.eval import runner  # noqa: E402
from gal3uwb.filters.common import FilterNoise  # noqa: E402
from gal3uwb.filters.ekf import Ekf  # noqa: E402
from gal3uwb.filters.eqf import Eqf  # noqa: E402
from gal3uwb.params import ImuNoise  # noqa: E402
from gal3uwb.sim.scenario import SCENARIOS  # noqa: E402

from ekf_fej import EkfFej  # noqa: E402
from ekf_slam import EkfSlam  # noqa: E402
from eqf_slam import EqfSlam  # noqa: E402
from model import SIGMA_ANCHOR, SlamEstimate, initial_estimate, prior_covariance, run_filter, summarize  # noqa: E402

FLIGHTS = [
    "default_1_random3_0", "default_1_random3_1", "default_1_random3_2", "default_1_circular2D_0",
    "default_1_circular3D_0", "default_1_remoteControlLowPace_0", "default_1_remoteControlLowPace_0_v2",
    "default_1_remoteControlHighPace_0", "default_1_remoteControlHighPace_1", "default_1_remoteControlHighPace_2",
]
SLAM = {"EKF": EkfSlam, "FEJ-EKF": EkfFej, "EqF": EqfSlam}
SURVEYED = {"EKF surveyed": Ekf, "EqF surveyed": Eqf}
UWB_RATE, SIGMA_RANGE = 20.0, 0.1
WARMUP = 10.0  # [s]
RECORD_EVERY = 25  # IMU steps between records (10 Hz at 250 Hz)
# Gyro bias calibrated at rest; MILUV's PX4 shows ~0.5 m/s^2 of apparent accel bias (bias plus mounting tilt).
PRIOR = {**runner.INITIAL_SIGMA, "pos": 0.1, "gyro_bias": 0.005, "accel_bias": 0.5}
METRICS = ("pos_rmse", "pos_h_rmse", "pos_v_rmse", "yaw_rmse_deg", "offset_rmse_m", "anchor_rmse", "anchor_h_rmse",
           "nees_pvc", "nees_anchor_pos")

_FLIGHTS = {}


def prepare(name):
    """Load a flight, fit the mocap splines and estimate biases and IMU noise once."""
    flight = Flight.load(name)
    flight.traj.position(flight.t_imu[:2])
    flight.traj.rotation(flight.t_imu[:2])
    gyro_nd, accel_nd = flight.vibration_noise()
    static = imu_calibration(flight.robot)
    noise = ImuNoise(gyro_nd, accel_nd, 10 * static.gyro_bias_rw, 10 * static.accel_bias_rw)
    return name, {"flight": flight, "biases": flight.bias_estimate(), "imu_noise": noise,
                  "speed": flight.speed_stats(), "gyro_bias0": flight.static_gyro_bias()}


def _load_cache(path):
    global _FLIGHTS
    with open(path, "rb") as f:
        _FLIGHTS = pickle.load(f)


def run_task(task):
    name, scenario, seed, sigma_anchor, nu = task
    entry = _FLIGHTS[name]
    sim = simulate_on_flight(entry["flight"], SCENARIOS[scenario], UWB_RATE, SIGMA_RANGE, seed, biases=entry["biases"])
    sim = replace(sim, imu_noise=entry["imu_noise"])
    n, N = sim.truth.anchor_offset.shape[1], len(sim.uwb.anchors)
    P0 = prior_covariance(PRIOR, n, N, sigma_anchor)
    truth0 = SlamEstimate(runner.truth_estimate(sim.truth, 0), sim.uwb.anchors)
    x0 = initial_estimate(truth0, P0, np.random.default_rng(seed + 1_000_000))
    x0 = SlamEstimate(replace(x0.nav, gyro_bias=entry["gyro_bias0"]), x0.anchors)
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, SIGMA_RANGE, nu_sigma0=nu)
    out = {}
    for fname, cls in SLAM.items():
        run = run_filter(sim, cls, x0, P0, noise, RECORD_EVERY)
        out[fname] = summarize(run, WARMUP, n, N)
    k = len(P0) - 3 * N
    for fname, cls in SURVEYED.items():
        m = runner.summarize_run(runner.run_filter(sim, cls, x0.nav, P0[:k, :k], noise, record_every=RECORD_EVERY),
                                 WARMUP)
        out[fname] = {key: m[key] for key in METRICS if key in m}
    return name, scenario, seed, out


def summarize_table(saved):
    table = []
    for name in saved["flights"]:
        for scenario in saved["scenarios"]:
            for fname in list(SLAM) + list(SURVEYED):
                ms = [r[3][fname] for r in saved["results"] if r[0] == name and r[1] == scenario]
                row = {"flight": name, "scenario": scenario, "filter": fname, "speed_mean": saved["speed"][name][0]}
                for m in METRICS:
                    vals = np.array([x.get(m, np.nan) for x in ms])
                    row[m] = np.sqrt(np.mean(vals**2)) if "rmse" in m else np.mean(vals)
                table.append(row)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--flights", nargs="+", default=FLIGHTS)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--sigma-anchor", type=float, default=SIGMA_ANCHOR)
    parser.add_argument("--nu", type=float, default=3e-3, help="prior std of the EqF's virtual bias b_nu [m/s]")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/01_unsurveyed_anchors/miluv"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "flights.pkl"

    t0 = time.time()
    with Pool(min(args.workers, len(args.flights))) as pool:
        flights = dict(pool.map(prepare, args.flights))
    with open(cache, "wb") as f:
        pickle.dump(flights, f)
    print(f"prepared {len(flights)} flights in {time.time() - t0:.0f} s", flush=True)
    tasks = [(n, s, seed, args.sigma_anchor, args.nu) for n in args.flights for s in args.scenarios
             for seed in range(args.seeds)]
    print(f"{len(tasks)} runs, anchor prior {args.sigma_anchor:g} m, {args.workers} workers", flush=True)
    results = []
    with Pool(args.workers, initializer=_load_cache, initargs=(cache,)) as pool:
        for done, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            results.append(r)
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs ({time.time() - t0:.0f} s)", flush=True)
    saved = {"results": results, "speed": {n: e["speed"] for n, e in flights.items()}, "flights": args.flights,
             "scenarios": args.scenarios, "seeds": args.seeds, "sigma_anchor": args.sigma_anchor}
    with open(args.out / "runs.pkl", "wb") as f:
        pickle.dump(saved, f)
    cache.unlink()

    table = summarize_table(saved)
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(table[0]))
        w.writeheader()
        for row in table:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\n{'flight':>36} {'scenario':>13} {'filter':>13} | {'h cm':>5} {'z cm':>5} {'anch cm':>7} | "
          f"{'pvc':>6} {'anchors':>7}")
    for r in table:
        print(f"{r['flight']:>36} {r['scenario']:>13} {r['filter']:>13} | {100 * r['pos_h_rmse']:5.1f} "
              f"{100 * r['pos_v_rmse']:5.1f} {100 * r['anchor_rmse']:7.1f} | {r['nees_pvc']:6.2f} "
              f"{r['nees_anchor_pos']:7.2f}")
    print(f"\nwrote {args.out}/summary.csv, runs.pkl")


if __name__ == "__main__":
    main()
