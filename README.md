# gal3uwb

Galilean equivariant filtering for listen-only (TDoA) UWB + IMU localization.
The proposal and design discussion are in [`doc/`](doc/).

## Setup

```bash
uv sync            # creates .venv with gtsam 4.3.0 (PyPI wheel, includes Gal3)
uv run pytest      # smoke tests: GTSAM install and Gal3 conventions
```

## Layout

```
src/gal3uwb/
  sim/        trajectories, IMU, device clocks, UWB timestamps
  filters/    Galilean EqF and EKF baselines
  eval/       Monte Carlo harness, RMSE / NEES
experiments/  runnable scripts, one per study
tests/
results/      generated outputs (gitignored)
```

## Experiments

- `experiments/coupling_check.py`: paired Monte Carlo test of whether the
  Galilean clock-rate coupling matters for UWB. Two filters that differ only in
  integrating the IMU over true time vs. tag time, swept over speed and skew.
  Writes `results/coupling_check/{summary.csv,runs.npz,coupling_check.png}`;
  `--replot` redraws from saved results.

## Simulator conventions

- Tag time `T`, true (network) time `t`, offset `tau = T - t`, skew `dtau/dT`.
  Filters run on `T`; clock states are in metres (`beta = C tau`, `gamma = C skew`).
- Anchors are synchronized to true time by default (per-anchor offsets/skews are
  supported but zero). IMU and UWB share the tag clock.

## Gal3 conventions in GTSAM

Checked in `tests/test_environment.py`:

- Constructor: `Gal3(R, p, v, t)`, with position before velocity.
- Tangent vector (Expmap/Logmap): `[omega(0:3), v(3:6), p(6:9), t(9)]`, with velocity before position.
- Matrix: `[[R, v, p], [0, 1, t], [0, 0, 1]]`.
