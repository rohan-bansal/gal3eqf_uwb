# gal3uwb

Galilean equivariant filtering for UWB + IMU drone localization, where the tag's
clock offset is the Galilean time coordinate and its skew is the time-rate bias.
The proposal and design discussion are in [`doc/`](doc/).

The current setup follows the plan agreed after the coupling check (commit
`953f5b6`): the tag on the drone transmits and every anchor timestamps the
arrival. Steps 1 and 2 use synchronized anchors; step 3 gives every anchor its
own clock and has the anchors exchange packets so their clocks stay observable.

## Setup

```bash
uv sync                                   # .venv with gtsam 4.3.0 (PyPI wheel, includes Gal3)
uv run pytest                             # Lie group, Jacobian and simulator tests
python scripts/download_miluv.py          # MILUV sensor CSVs only (~0.74 GB of 186 GB)
```

Run experiments through Slurm (`slurm/experiment.sbatch`), not on the login node.

## Layout

```
src/gal3uwb/
  gal3.py     Gal(3) and its tangent group Gal(3) x| gal(3)
  sim/        trajectories, IMU, tag clock, tag-transmit UWB with synchronized anchors
  filters/    EqF on Gal(3) x| gal(3) (Delama et al.) and the SE2(3) EKF baseline
  data/       MILUV loader: real path and IMU with simulated UWB on top
  eval/       filter runner, RMSE and NEES
experiments/  one numbered folder per study, each with its own README
scripts/      data download
tests/
results/      generated outputs (gitignored)
data/         downloaded datasets (gitignored)
```

## Experiments

Each study lives in its own folder under `experiments/`, numbered in the order
they were done, with a README that says what it tests, how to run it and where
its results go. Code that only one study needs stays in its folder; the shared
library (groups, simulator, baseline filters, runner) is `src/gal3uwb/`.

- `experiments/00_eqf_vs_ekf/`: the EqF on Gal(3) against the EKF with surveyed
  anchors, for three clock scenarios (synchronized anchors with a fixed or a
  drifting tag clock; free-running anchors with anchor-to-anchor packets), on
  synthetic circles from hover to 45 m/s and on ten MILUV flights.
- `experiments/01_unsurveyed_anchors/`: the anchor positions are unknown and
  estimated with the drone. An EKF with additive anchor states, the same EKF
  with first-estimate Jacobians (the standard consistency fix), and an EqF
  whose symmetry group also moves the anchors.

Each writes a `summary.csv`, the raw runs and figures under `results/`.

## Conventions

- Tag time `T`, network time `t`. The filters estimate `delta = t - T` [s] and the
  time-rate bias `rho` (the tag skew), with `d(delta)/dT = -rho`.
- A transmission at tag time `T_tx` reaches anchor `i` at network time
  `t_rx,i`; the filters use `y_i = C (t_rx,i - T_tx) = C delta + |p_tag - a_i|`.
- EqF state: `F = (R, v_F, p_F, delta)` in Gal(3) is the drone frame relative to
  the Earth frame at the same network time, so `v = v_F + g delta` and
  `p = p_F + g delta^2 / 2`. Biases `(b_omega, b_a, b_nu, b_rho)` live in gal(3);
  `b_nu` is virtual (zero in truth).
- Step 3: anchor 0 is the network reference, so `delta = A_0 - T`; the other
  anchors carry `Delta_i = A_i - A_0` and skews `kappa_i` as additive states (a
  direct product with Gal(3) x| gal(3) in the EqF). Their noise is correlated
  through anchor 0's clock. Anchor-to-anchor packets give
  `C (A_j,rx - A_i,tx) = C (Delta_j - Delta_i) + |a_j - a_i|`.
- Both filters are compared in the same physical error coordinates
  (`filters/common.py`); "native" NEES uses each filter's own error definition.

GTSAM's Gal3 conventions (checked in `tests/test_environment.py`): constructor
`Gal3(R, p, v, t)`; tangent `[omega, v, p, t]`; matrix `[[R, v, p], [0, 1, t], [0, 0, 1]]`.
