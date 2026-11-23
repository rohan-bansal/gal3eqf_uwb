"""Sensitivity of the EqF to its virtual velocity-input bias b_nu, and to initialization.

b_nu is zero in truth, but in the EqF's error coordinates it absorbs second-order
terms like (position error) x (gyro bias error). A prior much tighter than that
product makes the EqF inconsistent; a loose one adds a nuisance direction. This
sweeps the b_nu prior for a pessimistic initialization (0.3 m, 0.01 rad/s) and a
first-fix initialization (0.1 m, gyro bias calibrated at rest to 0.002 rad/s).
Step 2 clock model, synthetic circle flights, EKF alongside as the reference.

    sbatch --cpus-per-task=8 slurm/experiment.sbatch experiments/00_eqf_vs_ekf/eqf_tuning.py
"""

import argparse
import csv
import os
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

from gal3uwb.eval.runner import INITIAL_SIGMA, initial_estimate, prior_covariance, run_filter, summarize_run, truth_estimate
from gal3uwb.filters.common import FilterNoise
from gal3uwb.filters.ekf import Ekf
from gal3uwb.filters.eqf import Eqf
from gal3uwb.sim.scenario import SCENARIOS, SimConfig, simulate

SPEEDS = [0.0, 0.5, 2.0]
INITS = {"pessimistic": (0.3, 0.01), "first_fix": (0.1, 0.002)}  # (position std [m], gyro bias std [rad/s])
NU_SIGMAS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
METRICS = ("pos_rmse", "vel_rmse", "yaw_rmse_deg", "nees_native", "nees_full", "nees_yaw", "nees_vel", "nees_pos",
           "nees_bias")


def run_task(task):
    init, speed, seed, duration = task
    pos_sigma, gyro_sigma = INITS[init]
    config = SimConfig(speed=speed, clock=SCENARIOS["gaussian_skew"], duration=duration, gyro_bias_sigma=gyro_sigma)
    sim = simulate(config, seed)
    P0 = prior_covariance({**INITIAL_SIGMA, "pos": pos_sigma, "gyro_bias": gyro_sigma})
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(seed + 1_000_000))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, config.sigma_range)
    out = {"EKF": summarize_run(run_filter(sim, Ekf, x0, P0, noise, record_every=20), 5.0)}
    for nu in NU_SIGMAS:
        out[f"EqF nu={nu:g}"] = summarize_run(run_filter(sim, Eqf, x0, P0, replace(noise, nu_sigma0=nu), record_every=20), 5.0)
    return init, speed, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/eqf_tuning"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(i, v, s, args.duration) for i in INITS for v in SPEEDS for s in range(args.runs)]
    results = []
    with Pool(args.workers) as pool:
        for done, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            results.append(r)
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs", flush=True)
    rows = []
    for init in INITS:
        for speed in SPEEDS:
            outs = [r[2] for r in results if r[0] == init and r[1] == speed]
            for name in outs[0]:
                row = {"init": init, "speed": speed, "filter": name}
                for m in METRICS:
                    vals = np.array([o[name][m] for o in outs])
                    row[m] = float(np.sqrt(np.mean(vals**2)) if "rmse" in m else np.mean(vals))
                rows.append(row)
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'init':>11} {'v':>4} {'filter':>14} | {'pos cm':>6} {'vel cm/s':>8} {'yaw deg':>7} | "
          f"{'native':>6} {'phys':>5} {'yaw':>5} {'vel':>5} {'pos':>5} {'bias':>5}")
    for r in rows:
        print(f"{r['init']:>11} {r['speed']:4.1f} {r['filter']:>14} | {100 * r['pos_rmse']:6.2f} "
              f"{100 * r['vel_rmse']:8.2f} {r['yaw_rmse_deg']:7.2f} | {r['nees_native']:6.2f} {r['nees_full']:5.2f} "
              f"{r['nees_yaw']:5.2f} {r['nees_vel']:5.2f} {r['nees_pos']:5.2f} {r['nees_bias']:5.2f}")


if __name__ == "__main__":
    main()
