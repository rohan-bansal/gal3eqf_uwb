# gal3uwb

Galilean equivariant filtering for UWB + IMU drone localization, where the tag's
clock offset is the Galilean time coordinate and its skew is the time-rate bias.
The proposal and design discussion are in [`doc/`](doc/).

The current setup follows the plan agreed after the coupling check (commit
`953f5b6`): the tag on the drone transmits and every anchor timestamps the
arrival. Steps 1 and 2 use synchronized anchors.

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
experiments/  runnable studies
scripts/      data download
tests/
results/      generated outputs (gitignored)
data/         downloaded datasets (gitignored)
```

## Experiments

- `experiments/synced_anchors.py`: steps 1 and 2 on synthetic circle flights at
  0, 0.5, 2 and 5 m/s. Step 1 is an ideal tag clock (unknown constant offset);
  step 2 draws the skew from N(0, 20 ppm^2) per flight and adds TCXO noise.
  Both filters run on the same data from the same initial estimate.
- `experiments/miluv_synced.py`: the same comparison on ten single-drone MILUV
  flights, with the real mocap path and PX4 IMU and simulated UWB at MILUV's
  anchor and tag positions.
- `experiments/eqf_tuning.py`: sensitivity of the EqF to the prior on its
  virtual velocity-input bias and to the initial uncertainty.

Each writes a `summary.csv`, the raw runs and figures under `results/<name>/`.

## Conventions

- Tag time `T`, network time `t`. The filters estimate `delta = t - T` [s] and the
  time-rate bias `rho` (the tag skew), with `d(delta)/dT = -rho`.
- A transmission at tag time `T_tx` reaches anchor `i` at network time
  `t_rx,i`; the filters use `y_i = C (t_rx,i - T_tx) = C delta + |p_tag - a_i|`.
- EqF state: `F = (R, v_F, p_F, delta)` in Gal(3) is the drone frame relative to
  the Earth frame at the same network time, so `v = v_F + g delta` and
  `p = p_F + g delta^2 / 2`. Biases `(b_omega, b_a, b_nu, b_rho)` live in gal(3);
  `b_nu` is virtual (zero in truth).
- Both filters are compared in the same physical error coordinates
  (`filters/common.py`); "native" NEES uses each filter's own error definition.

GTSAM's Gal3 conventions (checked in `tests/test_environment.py`): constructor
`Gal3(R, p, v, t)`; tangent `[omega, v, p, t]`; matrix `[[R, v, p], [0, 1, t], [0, 0, 1]]`.
