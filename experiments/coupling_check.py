"""Does the Galilean clock-rate coupling matter for UWB? Paired Monte Carlo.

For each (speed, skew, seed), simulate one flight and run the coupled and
decoupled filters on the same data from the same initial estimate. The only
difference between them is whether IMU integration uses true time
dT (1 - gamma / C) or tag time dT (see gal3uwb.filters.eskf).

    uv run python experiments/coupling_check.py              # full sweep
    uv run python experiments/coupling_check.py --runs 4     # quick look
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

# One BLAS thread per worker. Threaded BLAS on the filter's 17x17 matrices is ~40x
# slower, and Slurm here sets these to the allocated CPU count, so override them.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from gal3uwb.eval.runner import initial_estimate, run_filter
from gal3uwb.filters.eskf import POS
from gal3uwb.sim.scenario import SimConfig, simulate

SPEEDS = [0.0, 0.5, 2.0, 5.0]  # [m/s]
SKEWS = [1e-5, 4e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # tag clock skew magnitude [-]
REALISTIC_SKEW = (1e-5, 4e-5)  # +/-20 ppm crystals at both ends: up to 40 ppm relative
WARMUP = 5.0  # discard the convergence transient [s]
FILTERS = {"decoupled": False, "coupled": True}
METRICS = ("pos_sq", "nees", "nees_pos", "nees_clock")
DOF = {"nees": 17, "nees_pos": 3, "nees_clock": 2}


def run_task(task):
    i_speed, i_skew, seed, duration = task
    sim = simulate(SimConfig(speed=SPEEDS[i_speed], skew=SKEWS[i_skew], duration=duration), seed)
    x0, P0 = initial_estimate(sim, np.random.default_rng(seed + 1_000_000))
    runs = {name: run_filter(sim, coupled, x0, P0) for name, coupled in FILTERS.items()}
    keep = runs["coupled"].T >= WARMUP
    metrics = {
        name: {
            "pos_sq": np.sum(r.err[keep][:, POS] ** 2, axis=1).mean(),
            "nees": r.nees[keep].mean(),
            "nees_pos": r.nees_pos[keep].mean(),
            "nees_clock": r.nees_clock[keep].mean(),
        }
        for name, r in runs.items()
    }
    diff = np.linalg.norm(runs["decoupled"].p_hat - runs["coupled"].p_hat, axis=1)[keep]
    return i_speed, i_skew, seed, metrics, np.sqrt(np.mean(diff**2)), diff.max()


def sweep(n_runs, duration, workers):
    shape = (len(SPEEDS), len(SKEWS), n_runs)
    res = {f"{name}_{m}": np.zeros(shape) for name in FILTERS for m in METRICS}
    res["diff_rms"], res["diff_max"] = np.zeros(shape), np.zeros(shape)
    tasks = [(i, j, s, duration) for i in range(len(SPEEDS)) for j in range(len(SKEWS)) for s in range(n_runs)]
    with Pool(workers) as pool:
        for done, (i, j, s, metrics, d_rms, d_max) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            for name in FILTERS:
                for m in METRICS:
                    res[f"{name}_{m}"][i, j, s] = metrics[name][m]
            res["diff_rms"][i, j, s], res["diff_max"][i, j, s] = d_rms, d_max
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs", flush=True)
    return res


def summarize(res):
    """Average over runs: RMSE, NEES per degree of freedom, filter difference."""
    out = {}
    for name in FILTERS:
        out[f"{name}_pos_rmse"] = np.sqrt(res[f"{name}_pos_sq"].mean(-1))
        for m, dof in DOF.items():
            out[f"{name}_{m}_per_dof"] = res[f"{name}_{m}"].mean(-1) / dof
    out["diff_rms"] = np.sqrt((res["diff_rms"] ** 2).mean(-1))
    out["diff_max"] = res["diff_max"].max(-1)
    return out


def write_csv(summary, path):
    keys = list(summary)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed_mps", "skew"] + keys)
        for i, speed in enumerate(SPEEDS):
            for j, skew in enumerate(SKEWS):
                w.writerow([speed, skew] + [f"{summary[k][i, j]:.6g}" for k in keys])


# Ordinal blue ramp (validated light->dark) for speed; line style for filter.
SPEED_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def plot(summary, n_runs, path):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    skews = np.array(SKEWS)
    lo, hi = chi2.ppf([0.025, 0.975], n_runs * DOF["nees_pos"]) / (n_runs * DOF["nees_pos"])

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.axvspan(*REALISTIC_SKEW, color=GRID, alpha=0.7, linewidth=0)
        ax.set_xlabel("tag clock skew [ppm]")
        ax.set_xticks(skews, [f"{s * 1e6:,.0f}" for s in skews])
        ax.minorticks_off()

    ax = axes[0]
    for i, speed in enumerate(SPEEDS):
        y = np.maximum(summary["diff_rms"][i], 1e-9)
        ax.plot(skews, y, "-o", color=SPEED_COLORS[i], lw=2, ms=6, label=f"{speed:g} m/s")
        ax.annotate(f"{speed:g} m/s", (skews[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    rmse = summary["coupled_pos_rmse"].mean()
    ax.axhline(rmse, color=INK_2, lw=1, ls=":")
    ax.text(skews[0], rmse * 1.3, f"position error itself (~{100 * rmse:.0f} cm)", color=INK_2, fontsize=8)
    ax.set_ylabel("RMS gap between filters' positions [m]")
    ax.set_title("How much does the coupling move the estimate?", loc="left", color=INK, fontsize=10)
    ax.set_xlim(skews[0] / 1.5, skews[-1] * 4)

    for ax, key, ylabel, title in (
        (axes[1], "pos_rmse", "position RMSE [m]", "Accuracy"),
        (axes[2], "nees_pos_per_dof", "position NEES / dof (1 = consistent)", "Consistency"),
    ):
        for i, speed in enumerate(SPEEDS):
            for name, ls, mfc in (("coupled", "-", SPEED_COLORS[i]), ("decoupled", "--", SURFACE)):
                ax.plot(skews, summary[f"{name}_{key}"][i], ls=ls, marker="o", color=SPEED_COLORS[i],
                        mfc=mfc, lw=2, ms=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[2].axhspan(lo, hi, color="#cde2fb", alpha=0.6, linewidth=0)

    speed_handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=6) for c in SPEED_COLORS]
    style_handles = [
        plt.Line2D([], [], color=INK_2, lw=2, ls="-", marker="o", ms=6),
        plt.Line2D([], [], color=INK_2, lw=2, ls="--", marker="o", ms=6, mfc=SURFACE),
        plt.Rectangle((0, 0), 1, 1, color=GRID),
    ]
    fig.legend(
        speed_handles + style_handles,
        [f"{s:g} m/s" for s in SPEEDS] + ["coupled (Galilean time rate)", "decoupled (clock beside pose)", "realistic crystals"],
        loc="lower center", ncol=7, frameon=False, fontsize=8, labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs per (speed, skew)")
    parser.add_argument("--duration", type=float, default=30.0, help="flight duration [s]")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/coupling_check"))
    parser.add_argument("--replot", action="store_true", help="re-plot from <out>/runs.npz without running")
    args = parser.parse_args()

    if args.replot:
        with np.load(args.out / "runs.npz") as data:
            res = {k: data[k] for k in data.files if k not in ("speeds", "skews")}
        n_runs = res["diff_rms"].shape[-1]
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        n_tasks = len(SPEEDS) * len(SKEWS) * args.runs
        print(f"{n_tasks} paired runs on {args.workers} workers")
        t0 = time.time()
        res = sweep(args.runs, args.duration, args.workers)
        print(f"done in {time.time() - t0:.0f} s")
        np.savez(args.out / "runs.npz", speeds=SPEEDS, skews=SKEWS, **res)
        n_runs = args.runs

    summary = summarize(res)
    write_csv(summary, args.out / "summary.csv")
    plot(summary, n_runs, args.out / "coupling_check.png")

    print(f"\n{'speed':>6} {'skew':>7} | {'gap RMS [m]':>11} {'gap max [m]':>11} | "
          f"{'RMSE cpl':>8} {'RMSE dec':>8} | {'NEESp cpl':>9} {'NEESp dec':>9}")
    for i, speed in enumerate(SPEEDS):
        for j, skew in enumerate(SKEWS):
            print(
                f"{speed:6.1f} {skew:7.0e} | {summary['diff_rms'][i, j]:11.2e} {summary['diff_max'][i, j]:11.2e} | "
                f"{summary['coupled_pos_rmse'][i, j]:8.4f} {summary['decoupled_pos_rmse'][i, j]:8.4f} | "
                f"{summary['coupled_nees_pos_per_dof'][i, j]:9.2f} {summary['decoupled_nees_pos_per_dof'][i, j]:9.2f}"
            )
    print(f"\nwrote {args.out}/summary.csv, runs.npz, coupling_check.png")


if __name__ == "__main__":
    main()
