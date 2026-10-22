"""Steps 1 and 2: EqF on Gal(3) vs EKF with synchronized anchors, synthetic flights.

Step 1 ("ideal_clock"): unknown constant tag offset, no skew.
Step 2 ("gaussian_skew"): tag skew ~ N(0, 20 ppm^2) per flight, plus TCXO noise.
For each (scenario, speed, seed), simulate one flight and run every filter on
the same data from the same initial estimate. The default prior is a first UWB
fix (0.1 m) with the gyro bias calibrated at rest; --pos-sigma 0.3
--gyro-bias-sigma 0.01 gives the pessimistic stress case.

    uv run python experiments/synced_anchors.py --runs 4     # quick look
    sbatch slurm/experiment.sbatch experiments/synced_anchors.py
"""

import argparse
import csv
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

# One BLAS thread per worker; Slurm sets these to the allocated CPU count.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from gal3uwb.eval.runner import INITIAL_SIGMA, initial_estimate, prior_covariance, run_filter, summarize_run, truth_estimate
from gal3uwb.filters.common import FilterNoise
from gal3uwb.filters.ekf import Ekf
from gal3uwb.filters.eqf import Eqf
from gal3uwb.sim.scenario import SCENARIOS, SimConfig, simulate

SPEEDS = [0.0, 0.5, 2.0, 5.0]  # [m/s]
# name -> (filter class, FilterNoise overrides). The EqF runs with Delama's tight prior on
# its virtual bias b_nu and with a loose one (see experiments/eqf_tuning.py).
FILTERS = {"EKF": (Ekf, {}), "EqF": (Eqf, {"nu_sigma0": 1e-4}), "EqF-loose": (Eqf, {"nu_sigma0": 1e-2})}
FILTER_LABELS = {"EKF": "EKF", "EqF": "EqF, b_nu prior 1e-4", "EqF-loose": "EqF, b_nu prior 1e-2"}
SCENARIO_LABELS = {"ideal_clock": "step 1: ideal clock", "gaussian_skew": "step 2: Gaussian skew"}
WARMUP = 5.0  # discard the convergence transient [s]
RECORD_EVERY = 20  # IMU steps between records (10 Hz)
METRICS = ("pos_rmse", "vel_rmse", "rot_rmse_deg", "yaw_rmse_deg", "offset_rmse_m", "rho_rmse_ppm", "nees_full",
           "nees_native", "nees_rot", "nees_yaw", "nees_vel", "nees_pos", "nees_bias", "nees_clock")


def run_task(task):
    scenario, i_speed, seed, duration, skew_sigma, pos_sigma, gyro_bias_sigma, accel_bias_sigma = task
    clock = SCENARIOS[scenario]
    if scenario == "gaussian_skew" and skew_sigma is not None:
        clock = replace(clock, skew_sigma=skew_sigma)
    config = SimConfig(speed=SPEEDS[i_speed], clock=clock, duration=duration, gyro_bias_sigma=gyro_bias_sigma,
                       accel_bias_sigma=accel_bias_sigma)
    sim = simulate(config, seed)
    # The prior matches the truth's: bias spreads, and the oscillator spec for the skew
    # (no tighter than 20 ppm, so step 1's zero skew is not known in advance).
    P0 = prior_covariance({**INITIAL_SIGMA, "pos": pos_sigma, "gyro_bias": gyro_bias_sigma,
                           "accel_bias": accel_bias_sigma, "rho": max(INITIAL_SIGMA["rho"], clock.skew_sigma)})
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(seed + 1_000_000))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, config.sigma_range)
    out = {}
    for name, (cls, overrides) in FILTERS.items():
        run = run_filter(sim, cls, x0, P0, replace(noise, **overrides), record_every=RECORD_EVERY)
        out[name] = {
            "metrics": summarize_run(run, WARMUP),
            "series": {"nees_native": run.nees_native / 17, "nees_full": run.nees["full"] / 17,
                       "nees_yaw": run.nees_yaw},
            "T": run.T,
        }
    return scenario, i_speed, seed, out


def sweep(scenarios, n_runs, duration, workers, *sigmas):
    tasks = [(s, i, seed, duration, *sigmas) for s in scenarios for i in range(len(SPEEDS)) for seed in range(n_runs)]
    shape = (len(scenarios), len(SPEEDS), n_runs)
    res, T = {}, None
    with Pool(workers) as pool:
        for done, (scenario, i, seed, out) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            si = scenarios.index(scenario)
            for name, r in out.items():
                for m, v in r["metrics"].items():
                    res.setdefault(f"{name}_{m}", np.zeros(shape))[si, i, seed] = v
                for key, series in r["series"].items():
                    arr = res.setdefault(f"{name}_series_{key}", np.zeros(shape + (len(series),)))
                    arr[si, i, seed] = series
                T = r["T"]
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs", flush=True)
    res["T"] = T
    return res


def write_csv(res, scenarios, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "speed_mps", "filter"] + list(METRICS))
        for si, scenario in enumerate(scenarios):
            for i, speed in enumerate(SPEEDS):
                for name in FILTERS:
                    row = []
                    for m in METRICS:
                        v = res[f"{name}_{m}"][si, i]
                        # RMSEs combine as root-mean-square over runs; NEES as means.
                        row.append(np.sqrt(np.mean(v**2)) if "rmse" in m else np.mean(v))
                    w.writerow([scenario, speed, name] + [f"{x:.6g}" for x in row])


def print_table(res, scenarios):
    print(f"\n{'scenario':>14} {'speed':>5} {'filter':>9} | {'pos cm':>6} {'yaw deg':>7} {'offset cm':>9} | "
          f"{'NEES':>5} {'native':>6} {'rot':>5} {'yaw':>5} {'pos':>5} {'bias':>5} {'clock':>5}")
    for si, scenario in enumerate(scenarios):
        for i, speed in enumerate(SPEEDS):
            for name in FILTERS:
                def rms(m):
                    return np.sqrt(np.mean(res[f"{name}_{m}"][si, i] ** 2))

                def mean(m):
                    return np.mean(res[f"{name}_{m}"][si, i])

                print(f"{scenario:>14} {speed:5.1f} {name:>9} | {100 * rms('pos_rmse'):6.2f} {rms('yaw_rmse_deg'):7.3f} "
                      f"{100 * rms('offset_rmse_m'):9.2f} | {mean('nees_full'):5.2f} {mean('nees_native'):6.2f} "
                      f"{mean('nees_rot'):5.2f} {mean('nees_yaw'):5.2f} {mean('nees_pos'):5.2f} "
                      f"{mean('nees_bias'):5.2f} {mean('nees_clock'):5.2f}")


# Categorical slots 1-2 of the validated reference palette; text stays in ink.
FILTER_COLORS = {"EKF": "#2a78d6", "EqF": "#eb6834", "EqF-loose": "#1baf7a"}
INK, INK_2, SURFACE, GRID, BAND = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#ecebe7"


def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    })


def _axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)


def plot_summary(res, scenarios, n_runs, path):
    """Metrics vs speed: color is the filter, line style the scenario."""
    _style()
    panels = [
        ("pos_rmse", 100.0, "rms", "position RMSE [cm]", "Position"),
        ("yaw_rmse_deg", 1.0, "rms", "yaw RMSE [deg]", "Heading"),
        ("nees_native", 1.0, "mean", "NEES / dof, own error coordinates", "Full-state consistency"),
        ("nees_yaw", 1.0, "mean", "yaw NEES (1 = consistent)", "Heading consistency"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.0), facecolor=SURFACE)
    styles = {s: ("-", None) if k == 0 else ("--", SURFACE) for k, s in enumerate(scenarios)}
    for ax, (key, scale, agg, ylabel, title) in zip(axes, panels):
        _axes(ax)
        for si, scenario in enumerate(scenarios):
            ls, mfc = styles[scenario]
            for name, color in FILTER_COLORS.items():
                v = res[f"{name}_{key}"][si]
                y = np.sqrt(np.mean(v**2, axis=-1)) if agg == "rms" else np.mean(v, axis=-1)
                ax.plot(SPEEDS, scale * y, ls=ls, marker="o", color=color, mfc=mfc or color, lw=2, ms=6)
        if key.startswith("nees"):
            dof = 1 if key == "nees_yaw" else 17
            lo, hi = chi2.ppf([0.025, 0.975], n_runs * dof) / (n_runs * dof)
            ax.axhspan(lo, hi, color=BAND, linewidth=0, zorder=0)
            ax.axhline(1.0, color=INK_2, lw=1, ls=":")
        ax.set_xlabel("drone speed [m/s]")
        ax.set_xticks(SPEEDS)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=6) for c in FILTER_COLORS.values()]
    handles += [plt.Line2D([], [], color=INK_2, lw=2, ls=styles[s][0], marker="o", ms=6, mfc=styles[s][1] or INK_2)
                for s in scenarios]
    labels = [FILTER_LABELS[n] for n in FILTER_COLORS] + [SCENARIO_LABELS[s] for s in scenarios]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_series(res, scenarios, n_runs, path):
    """Average NEES across runs over time (ANEES), per speed, for each scenario."""
    _style()
    T = res["T"]
    rows = [(s, key) for s in scenarios for key in ("nees_native", "nees_yaw")]
    fig, axes = plt.subplots(len(rows), len(SPEEDS), figsize=(14, 2.3 * len(rows)), facecolor=SURFACE,
                             sharex=True, squeeze=False)
    for r, (scenario, key) in enumerate(rows):
        si = scenarios.index(scenario)
        dof = 1 if key == "nees_yaw" else 17
        lo, hi = chi2.ppf([0.025, 0.975], n_runs * dof) / (n_runs * dof)
        for c, speed in enumerate(SPEEDS):
            ax = axes[r, c]
            _axes(ax)
            ax.axhspan(lo, hi, color=BAND, linewidth=0, zorder=0)
            ax.axhline(1.0, color=INK_2, lw=1, ls=":")
            for name, color in FILTER_COLORS.items():
                ax.plot(T, res[f"{name}_series_{key}"][si, c].mean(axis=0), color=color, lw=1.5)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{speed:g} m/s", loc="left", color=INK, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{'NEES / dof' if dof == 17 else 'yaw NEES'}\n{SCENARIO_LABELS[scenario]}", fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel("time [s]")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in FILTER_COLORS.values()]
    fig.legend(handles, [FILTER_LABELS[n] for n in FILTER_COLORS], loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs per (scenario, speed)")
    parser.add_argument("--duration", type=float, default=30.0, help="flight duration [s]")
    parser.add_argument("--skew-sigma", type=float, default=None, help="override step 2's skew std [-]")
    parser.add_argument("--pos-sigma", type=float, default=0.1, help="initial position std [m] (first UWB fix)")
    parser.add_argument("--gyro-bias-sigma", type=float, default=0.002,
                        help="initial gyro bias std [rad/s] (calibrated at rest)")
    parser.add_argument("--accel-bias-sigma", type=float, default=0.05, help="initial accel bias std [m/s^2]")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/synced_anchors"))
    parser.add_argument("--replot", action="store_true", help="re-plot from <out>/runs.npz without running")
    args = parser.parse_args()

    if args.replot:
        with np.load(args.out / "runs.npz", allow_pickle=False) as data:
            res = {k: data[k] for k in data.files if k != "scenarios"}
            scenarios = [str(s) for s in data["scenarios"]]
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        scenarios = args.scenarios
        n_tasks = len(scenarios) * len(SPEEDS) * args.runs
        print(f"{n_tasks} flights x {len(FILTERS)} filters on {args.workers} workers", flush=True)
        t0 = time.time()
        res = sweep(scenarios, args.runs, args.duration, args.workers, args.skew_sigma, args.pos_sigma,
                    args.gyro_bias_sigma, args.accel_bias_sigma)
        print(f"done in {time.time() - t0:.0f} s")
        np.savez(args.out / "runs.npz", scenarios=np.array(scenarios), **res)

    n_runs = res["EKF_pos_rmse"].shape[-1]
    write_csv(res, scenarios, args.out / "summary.csv")
    plot_summary(res, scenarios, n_runs, args.out / "summary.png")
    plot_series(res, scenarios, n_runs, args.out / "nees_over_time.png")
    print_table(res, scenarios)
    print(f"\nwrote {args.out}/summary.csv, runs.npz, summary.png, nees_over_time.png")


if __name__ == "__main__":
    main()
