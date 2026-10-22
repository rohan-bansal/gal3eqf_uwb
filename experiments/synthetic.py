"""EqF on Gal(3) vs EKF on synthetic circle flights, for the three clock scenarios.

Step 1 ("ideal_clock"): synchronized anchors, unknown constant tag offset, no skew.
Step 2 ("gaussian_skew"): synchronized anchors, tag skew ~ N(0, 20 ppm^2), TCXO noise.
Step 3 ("unsynced"): every anchor runs its own TCXO clock; anchors exchange packets.
For each (scenario, speed, seed), simulate one flight and run every filter on
the same data from the same initial estimate. The default prior is a first UWB
fix (0.1 m) with the gyro bias calibrated at rest; --pos-sigma 0.3
--gyro-bias-sigma 0.01 gives the pessimistic stress case. Steps 1-2 in
results/synthetic_steps12* were run with --nu-loose 1e-2.

    uv run python experiments/synthetic.py --runs 4     # quick look
    sbatch slurm/experiment.sbatch experiments/synthetic.py --scenarios unsynced --out results/synthetic_step3
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

from gal3uwb.eval.runner import (INITIAL_SIGMA, initial_estimate, prior_covariance, run_filter, summarize_run,
                                 truth_estimate)
from gal3uwb.filters.common import FilterNoise
from gal3uwb.filters.ekf import Ekf
from gal3uwb.filters.eqf import Eqf
from gal3uwb.sim.scenario import SCENARIOS, SimConfig, simulate

SPEEDS = [0.0, 0.5, 2.0, 5.0]  # [m/s]
# The EqF runs with Delama's tight prior on its virtual bias b_nu and with a looser one
# (--nu-loose; experiments/eqf_tuning.py found ~3e-3 best).
FILTERS = {"EKF": Ekf, "EqF": Eqf, "EqF-loose": Eqf}
NU_TIGHT = 1e-4
SCENARIO_LABELS = {"ideal_clock": "step 1: ideal clock", "gaussian_skew": "step 2: Gaussian skew",
                   "unsynced": "step 3: unsynced anchors"}
WARMUP = 5.0  # discard the convergence transient [s]
RECORD_EVERY = 20  # IMU steps between records (10 Hz)
METRICS = ("pos_rmse", "vel_rmse", "rot_rmse_deg", "yaw_rmse_deg", "offset_rmse_m", "rho_rmse_ppm",
           "anchor_offset_rmse_m", "nees_full", "nees_native", "nees_rot", "nees_yaw", "nees_vel", "nees_pos",
           "nees_bias", "nees_clock", "nees_anchors")


def filter_labels(nu_loose):
    return {"EKF": "EKF", "EqF": f"EqF, b_nu prior {NU_TIGHT:g}", "EqF-loose": f"EqF, b_nu prior {nu_loose:g}"}


def present(res, metrics=METRICS):
    """Metrics recorded in this run; anchor-clock metrics exist only for step 3."""
    return [m for m in metrics if f"EKF_{m}" in res]


def run_task(task):
    scenario, i_speed, seed, duration, nu_loose, skew_sigma, pos_sigma, gyro_bias_sigma, accel_bias_sigma = task
    clock = SCENARIOS[scenario]
    if scenario == "gaussian_skew" and skew_sigma is not None:
        clock = replace(clock, skew_sigma=skew_sigma)
    config = SimConfig(speed=SPEEDS[i_speed], clock=clock, duration=duration, gyro_bias_sigma=gyro_bias_sigma,
                       accel_bias_sigma=accel_bias_sigma)
    sim = simulate(config, seed)
    # The prior matches the truth's: bias spreads, and the oscillator spec for the skew
    # (no tighter than 20 ppm, so step 1's zero skew is not known in advance).
    n = sim.truth.anchor_offset.shape[1]
    P0 = prior_covariance({**INITIAL_SIGMA, "pos": pos_sigma, "gyro_bias": gyro_bias_sigma,
                           "accel_bias": accel_bias_sigma, "rho": max(INITIAL_SIGMA["rho"], clock.skew_sigma)}, n)
    x0 = initial_estimate(truth_estimate(sim.truth, 0), P0, np.random.default_rng(seed + 1_000_000))
    noise = FilterNoise(sim.imu_noise, sim.clock_noise, config.sigma_range)
    nu = {"EKF": NU_TIGHT, "EqF": NU_TIGHT, "EqF-loose": nu_loose}
    out = {}
    for name, cls in FILTERS.items():
        run = run_filter(sim, cls, x0, P0, replace(noise, nu_sigma0=nu[name]), record_every=RECORD_EVERY)
        dof = run.err.shape[1]
        out[name] = {
            "dof": dof,
            "metrics": summarize_run(run, WARMUP),
            "series": {"nees_native": run.nees_native / dof, "nees_full": run.nees["full"] / dof,
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
                res.setdefault("dof", np.zeros(len(scenarios)))[si] = r["dof"]
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs", flush=True)
    res["T"] = T
    return res


def write_csv(res, scenarios, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        metrics = present(res)
        w.writerow(["scenario", "speed_mps", "filter"] + metrics)
        for si, scenario in enumerate(scenarios):
            for i, speed in enumerate(SPEEDS):
                for name in FILTERS:
                    row = []
                    for m in metrics:
                        v = res[f"{name}_{m}"][si, i]
                        # RMSEs combine as root-mean-square over runs; NEES as means.
                        row.append(np.sqrt(np.mean(v**2)) if "rmse" in m else np.mean(v))
                    w.writerow([scenario, speed, name] + [f"{x:.6g}" for x in row])


def print_table(res, scenarios):
    anchors = "nees_anchors" in present(res)
    print(f"\n{'scenario':>14} {'speed':>5} {'filter':>9} | {'pos cm':>6} {'yaw deg':>7} {'offset cm':>9} "
          + (f"{'anch cm':>7} " if anchors else "")
          + f"| {'NEES':>5} {'native':>6} {'rot':>5} {'yaw':>5} {'pos':>5} {'bias':>5} {'clock':>5}"
          + (f" {'anch':>5}" if anchors else ""))
    for si, scenario in enumerate(scenarios):
        for i, speed in enumerate(SPEEDS):
            for name in FILTERS:
                def rms(m):
                    return np.sqrt(np.mean(res[f"{name}_{m}"][si, i] ** 2))

                def mean(m):
                    return np.mean(res[f"{name}_{m}"][si, i])

                print(f"{scenario:>14} {speed:5.1f} {name:>9} | {100 * rms('pos_rmse'):6.2f} {rms('yaw_rmse_deg'):7.3f} "
                      f"{100 * rms('offset_rmse_m'):9.2f} "
                      + (f"{100 * rms('anchor_offset_rmse_m'):7.2f} " if anchors else "")
                      + f"| {mean('nees_full'):5.2f} {mean('nees_native'):6.2f} "
                      f"{mean('nees_rot'):5.2f} {mean('nees_yaw'):5.2f} {mean('nees_pos'):5.2f} "
                      f"{mean('nees_bias'):5.2f} {mean('nees_clock'):5.2f}"
                      + (f" {mean('nees_anchors'):5.2f}" if anchors else ""))


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


LINE_STYLES = [("-", None), ("--", SURFACE), (":", SURFACE)]  # per scenario: line style, marker fill


def plot_summary(res, scenarios, n_runs, labels, path):
    """Metrics vs speed: color is the filter, line style the scenario."""
    _style()
    panels = [
        ("pos_rmse", 100.0, "rms", "position RMSE [cm]", "Position"),
        ("yaw_rmse_deg", 1.0, "rms", "yaw RMSE [deg]", "Heading"),
        ("nees_native", 1.0, "mean", "NEES / dof, own error coordinates", "Full-state consistency"),
        ("nees_yaw", 1.0, "mean", "yaw NEES (1 = consistent)", "Heading consistency"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.0), facecolor=SURFACE)
    styles = dict(zip(scenarios, LINE_STYLES))
    for ax, (key, scale, agg, ylabel, title) in zip(axes, panels):
        _axes(ax)
        for si, scenario in enumerate(scenarios):
            ls, mfc = styles[scenario]
            for name, color in FILTER_COLORS.items():
                v = res[f"{name}_{key}"][si]
                y = np.sqrt(np.mean(v**2, axis=-1)) if agg == "rms" else np.mean(v, axis=-1)
                ax.plot(SPEEDS, scale * y, ls=ls, marker="o", color=color, mfc=mfc or color, lw=2, ms=6)
        if key.startswith("nees"):
            dof = 1 if key == "nees_yaw" else int(res["dof"].min())
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
    legend = [labels[n] for n in FILTER_COLORS] + [SCENARIO_LABELS[s] for s in scenarios]
    fig.legend(handles, legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_series(res, scenarios, n_runs, labels, path):
    """Average NEES across runs over time (ANEES), per speed, for each scenario."""
    _style()
    T = res["T"]
    rows = [(s, key) for s in scenarios for key in ("nees_native", "nees_yaw")]
    fig, axes = plt.subplots(len(rows), len(SPEEDS), figsize=(14, 2.3 * len(rows)), facecolor=SURFACE,
                             sharex=True, squeeze=False)
    for r, (scenario, key) in enumerate(rows):
        si = scenarios.index(scenario)
        dof = 1 if key == "nees_yaw" else int(res["dof"][si])
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
                ax.set_ylabel(f"{'yaw NEES' if key == 'nees_yaw' else 'NEES / dof'}\n{SCENARIO_LABELS[scenario]}",
                              fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel("time [s]")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in FILTER_COLORS.values()]
    fig.legend(handles, [labels[n] for n in FILTER_COLORS], loc="lower center", ncol=len(handles),
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
    parser.add_argument("--nu-loose", type=float, default=3e-3, help="b_nu prior of the second EqF [m/s]")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--replot", action="store_true", help="re-plot from <out>/runs.npz without running")
    args = parser.parse_args()

    if args.replot:
        with np.load(args.out / "runs.npz", allow_pickle=False) as data:
            res = {k: data[k] for k in data.files if k not in ("scenarios", "nu_loose")}
            scenarios = [str(s) for s in data["scenarios"]]
            nu_loose = float(data["nu_loose"]) if "nu_loose" in data.files else 1e-2  # older runs used 1e-2
            res.setdefault("dof", np.full(len(scenarios), 17.0))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        scenarios = args.scenarios
        n_tasks = len(scenarios) * len(SPEEDS) * args.runs
        print(f"{n_tasks} flights x {len(FILTERS)} filters on {args.workers} workers", flush=True)
        t0 = time.time()
        nu_loose = args.nu_loose
        res = sweep(scenarios, args.runs, args.duration, args.workers, nu_loose, args.skew_sigma, args.pos_sigma,
                    args.gyro_bias_sigma, args.accel_bias_sigma)
        print(f"done in {time.time() - t0:.0f} s")
        np.savez(args.out / "runs.npz", scenarios=np.array(scenarios), nu_loose=nu_loose, **res)

    n_runs = res["EKF_pos_rmse"].shape[-1]
    labels = filter_labels(nu_loose)
    write_csv(res, scenarios, args.out / "summary.csv")
    plot_summary(res, scenarios, n_runs, labels, args.out / "summary.png")
    plot_series(res, scenarios, n_runs, labels, args.out / "nees_over_time.png")
    print_table(res, scenarios)
    print(f"\nwrote {args.out}/summary.csv, runs.npz, summary.png, nees_over_time.png")


if __name__ == "__main__":
    main()
