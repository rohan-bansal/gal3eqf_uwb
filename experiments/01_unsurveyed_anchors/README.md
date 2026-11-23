# 01: Unsurveyed anchors

The anchor positions are not measured beforehand. Each filter starts from a
rough guess (each coordinate off by `SIGMA_ANCHOR`, 30 cm or 10 cm) and
estimates the anchor positions along with the drone, its biases and the clocks.

## Why this is a symmetry question

Rotating the whole scene (drone and anchors) about gravity, or translating it,
changes no range and no anchor-to-anchor distance. Those motions are
unobservable; only the priors (rough survey, takeoff pose) pin them down.

- `ekf_slam.py`: the EKF of experiment 00 with the anchor positions as additive
  states. In its coordinates the rotated direction is
  `(R^T e_z, e_z x v, e_z x p, e_z x a_i)`, which moves with every update, so
  the filter gains information it does not have (the EKF-SLAM inconsistency).
- `eqf_slam.py`: the EqF with the anchors inside the symmetry group,
  `(Gal(3) x| R^{3N}) x| gal(3)`. The anchors are extra columns of the Gal(3)
  matrix, as landmarks are in invariant EKF-SLAM. With the error
  `X = exp(eps) X_hat`, the rotation and translation of the scene are fixed
  vectors in `eps`, and propagation and every measurement leave them exactly
  unobservable (`test_slam.py` checks this, plus all Jacobians against finite
  differences).
- `ekf_fej.py`: the standard fix for EKF-SLAM inconsistency, first-estimate
  Jacobians (Huang, Mourikis & Roumeliotis 2008; Li & Mourikis 2013). The plain
  EKF's direction above is exactly unobservable at the point where its
  Jacobians are evaluated, but every update moves that point, so the next
  Jacobians are blind to a different direction and the old one leaks
  information. FEJ evaluates every Jacobian at points that never jump: the
  anchors at their first estimate (the survey guess) for the whole flight, the
  drone at its propagated estimate from before each update, with the step out of
  an update in Li & Mourikis' closed form. The linearized filter then has
  exactly the true unobservable directions, at the price of Jacobians at stale
  points. It is the fair baseline: the EqF gets the same property with Jacobians
  at the current estimate.

Anchor clocks (step 3) stay additive in all filters, with anchor 0 as the
reference, as in experiment 00.

## Files

| File | Contents |
|---|---|
| `model.py` | state with anchor positions, prior, run loop, metrics |
| `ekf_slam.py` | EKF with additive anchor states |
| `ekf_fej.py` | the same EKF with first-estimate Jacobians (FEJ-EKF) |
| `eqf_slam.py` | EqF on (Gal(3) x| R^{3N}) x| gal(3) |
| `test_slam.py` | Jacobian and unobservability checks, including FEJ's fixed gauge information (`.venv/bin/python -m pytest experiments/01_unsurveyed_anchors`) |
| `synthetic.py` | circle flights at 0, 0.5, 2, 5 m/s, three clock scenarios, 30 flights of 90 s each |
| `miluv.py` | ten MILUV flights, 8 draws each |

Both runners also run experiment 00's filters with the true anchor positions
as a reference ("surveyed").

    sbatch --cpus-per-task=32 slurm/experiment.sbatch experiments/01_unsurveyed_anchors/synthetic.py \
        --runs 30 --duration 90 --sigma-anchor 0.3 --out results/01_unsurveyed_anchors/synthetic_sa0.3
    sbatch --cpus-per-task=32 slurm/experiment.sbatch experiments/01_unsurveyed_anchors/miluv.py \
        --sigma-anchor 0.3 --out results/01_unsurveyed_anchors/miluv_sa0.3

## Results (synthetic, 30 flights of 90 s per setting)

Full-state NEES per degree of freedom (each filter in its own coordinates; 1
is honest), anchor-position NEES, and errors, averaged after a 5 s warm-up
(`results/01_unsurveyed_anchors/synthetic_sa{0.3,0.1}/summary.csv`). Each cell
is EKF / FEJ-EKF / EqF; position error adds the surveyed EqF.

30 cm survey error:

| step | m/s | NEES | anchor NEES | anchor err [cm] | pos err [cm] (+ surveyed) |
|---|---|---|---|---|---|
| 1 | 0 | 17.84 / 13.57 / 28.60 | 7.74 / 6.25 / 7.89 | 45.4 / 44.4 / 46.1 | 15.4 / 15.1 / 15.7 / 2.7 |
| 1 | 0.5 | 8.30 / 4.91 / 7.71 | 7.47 / 5.03 / 7.05 | 28.0 / 25.8 / 28.7 | 16.1 / 15.3 / 16.5 / 2.7 |
| 1 | 2 | 3.49 / 2.25 / 3.44 | 3.86 / 2.17 / 4.21 | 19.6 / 17.0 / 20.5 | 14.0 / 13.3 / 14.2 / 2.8 |
| 1 | 5 | 3.19 / 1.76 / **1.14** | 3.83 / 1.71 / **1.06** | 17.0 / 14.8 / 14.8 | 12.6 / 12.4 / 12.5 / 2.7 |
| 2 | 0 | 5.21 / 4.69 / 11.75 | 7.27 / 6.27 / 7.45 | 45.0 / 44.3 / 45.5 | 15.4 / 15.6 / 15.5 / 2.8 |
| 2 | 0.5 | 3.23 / 2.74 / 3.52 | 4.43 / 3.60 / 4.98 | 29.6 / 27.4 / 30.0 | 15.5 / 15.2 / 15.6 / 2.8 |
| 2 | 2 | 2.20 / 1.89 / 2.65 | 2.93 / 2.39 / 3.79 | 20.9 / 18.7 / 21.4 | 14.2 / 13.6 / 14.0 / 2.8 |
| 2 | 5 | 3.97 / **1.55** / 1.68 | 5.86 / 1.82 / 2.08 | 18.4 / 15.7 / 16.9 | 12.7 / 12.6 / 12.6 / 2.8 |
| 3 | 0 | 3.28 / 10.78 / 10.23 | 6.13 / 23.02 / 5.58 | 22.0 / 21.4 / 22.5 | 14.0 / 14.1 / 13.9 / 2.9 |
| 3 | 0.5 | 2.10 / 6.22 / 2.93 | 3.45 / 12.81 / 5.69 | 18.2 / 17.8 / 19.7 | 15.2 / 15.0 / 15.6 / 2.9 |
| 3 | 2 | 1.45 / 4.69 / **1.14** | 1.98 / 9.36 / 2.81 | 16.3 / 16.4 / 17.6 | 14.3 / 14.4 / 14.7 / 2.9 |
| 3 | 5 | 1.13 / 2.57 / **1.04** | 1.28 / 4.57 / 1.39 | 14.7 / 14.5 / 15.2 | 13.9 / 13.9 / 14.0 / 2.9 |

NEES at 90 s (30 cm, from `runs.npz`): step 1 at 5 m/s 5.75 / 1.95 / 1.05;
step 2 at 5 m/s 7.09 / 1.72 / 2.68; step 3 at 2 m/s 1.80 / 7.63 / 1.12.

- FEJ stops the EKF's drift in steps 1 and 2 but levels off above 1 (1.7 to
  2.5 at 2 and 5 m/s): its anchor Jacobians stay at the 30 cm-off guess.
- Brisk flight with a fixed tag clock (step 1, 5 m/s): only the EqF is honest.
  With a drifting tag clock (step 2) the EqF drifts to 2.7 by 90 s and FEJ is
  better.
- Anchor packets (step 3): FEJ is the most overconfident filter. The packets
  measure the anchor layout hundreds of times a second and FEJ applies that
  information along directions from the first guess. The EqF (flat at 1.04 to
  1.14 at 2 and 5 m/s) and the plain EKF (1.1 to 1.5, rising slowly) stay
  close; the EqF's anchor NEES is a little above the EKF's.
- Gentle flight (hover, 0.5 m/s, 2 m/s in steps 1 and 2): every filter's NEES
  grows. FEJ is the least bad in steps 1 and 2; the EqF is the worst at hover.
  The drone's own heading is barely observable then; that direction turns the
  drone alone, not the scene, and in the EqF's world-centred coordinates it
  depends on the estimate.
- Accuracy is close everywhere; FEJ's anchors are 2 to 3.5 cm closer at 0.5
  and 2 m/s in steps 1 and 2.
- With a 10 cm survey all three stay between 0.96 and 1.15 whenever the drone
  moves.

MILUV (`results/01_unsurveyed_anchors/miluv_sa{0.3,0.1}`): slow flights
(0.3 to 1.3 m/s) and nearly coplanar anchors put it in the gentle regime, and
every filter is overconfident about the anchors. Per-flight ranges, 30 cm
survey:

| step | horizontal err [cm] EKF / FEJ / EqF | anchor NEES EKF / FEJ / EqF |
|---|---|---|
| 1 | 12-21 / 12-(diverged) / 13-21 | 2.9-58 / 3.9-(diverged) / 3.0-33 |
| 2 | 14-80 / 12-18 / 14-50 | 9.2-723 / 8.2-52 / 8.3-399 |
| 3 | 14-16 / 13-16 / 14-16 | 8.0-27 / 18-49 / 11-34 |

FEJ is the most robust in step 2 and the most overconfident in step 3, as in
the synthetic gentle flights. It diverged (errors of metres) in 2 of 80 step-1
runs (`circular3D_0` seed 3, `remoteControlHighPace_0` seed 1); step 1 is where
even the surveyed filters are badly overconfident on MILUV, because the
no-drift clock model does not fit the real IMU. In the run we
traced, FEJ's NEES grows from about 9 s on, while the plain EKF's jumps only
at 27 s. With a 10 cm survey the horizontal error is 7
to 10 cm in steps 2 and 3 (about 4 cm surveyed) and anchor NEES 1.1 to 9.

Not tried: FEJ variants that refresh the first estimates once the anchors have
converged, or an observability-constrained EKF.

## Faster flights (30 cm survey)

In the default scene the circle radius is 2 m, so speed and turn acceleration
grow together (12.5 m/s^2 at 5 m/s). Three ways past 5 m/s, 30 flights of 90 s
each; the 5 m/s rows reproduce the table above exactly.

    # same room, harder turns (28 m/s^2 at 7.5 m/s, 50 m/s^2 at 10 m/s)
    sbatch slurm/experiment.sbatch experiments/01_unsurveyed_anchors/synthetic.py --speeds 5 7.5 10 \
        --runs 30 --duration 90 --out results/01_unsurveyed_anchors/fast_same_room_sa0.3
    # bigger scene, same 12.5 m/s^2 turn (scene x v^2: radius 8 / 18 / 32 m), survey error stays 30 cm
    sbatch slurm/experiment.sbatch experiments/01_unsurveyed_anchors/synthetic.py --speeds 5 10 15 20 \
        --turn-accel 12.5 --runs 30 --duration 90 --out results/01_unsurveyed_anchors/fast_big_scene_sa0.3
    # same, survey error scaled with the scene (1.2 / 2.7 / 4.8 m)
    ... --turn-accel 12.5 --scale-survey --out results/01_unsurveyed_anchors/fast_big_scene_scaled_sa0.3

Full-state NEES at 90 s, EKF / FEJ-EKF / EqF:

| setting | step 1 | step 2 | step 3 |
|---|---|---|---|
| room, 5 m/s | 5.75 / 1.95 / **1.05** | 7.09 / **1.72** / 2.68 | 1.21 / 3.72 / **1.04** |
| room, 7.5 m/s | 7.51 / **2.16** / 15.35 | 10.95 / **2.10** / 11.80 | 1.34 / 2.85 / **1.26** |
| room, 10 m/s | 3.98 / 4.90 / 8.95 | 6.86 / 5.20 / 10.29 | 3.61 / 4.97 / 3.51 |
| big scene, 10 m/s | 0.98 / 1.01 / 0.95 | 0.99 / 1.01 / 0.95 | 1.08 / 1.49 / 1.03 |
| big scene, 20 m/s | 0.97 / 0.97 / 0.94 | 1.01 / 1.00 / 0.97 | 1.23 / 1.26 / 1.21 |

- Same room: at 7.5 m/s the EqF loses its step-1 advantage (anchor NEES 10.3,
  anchors 19 cm off against FEJ's 15 cm); FEJ is best in steps 1 and 2. Not
  diagnosed yet. At 10 m/s (25 rad/s peak body rate) even the surveyed filters
  are at NEES 2.9 to 4.4: the 200 Hz IMU is too slow for that flight, so that
  row says little about the filters.
- Bigger scene: from 10 m/s on, every filter is consistent. A 30 cm survey in a
  scene 4 to 16 times larger is as easy as a 2 to 8 cm survey in the room.
- Scaled survey: every filter fails (NEES 28 to 5000); FEJ blows up in 3 to
  100 % of steps 1-2 flights (`failed` column). The second-order range error from
  the anchor guess, about |da|^2 / 2d, grows with the scene (about 2 cm in the
  room, 35 cm at 20 m/s) and swamps the 10 cm range noise; no single-linearization
  filter handles that without an iterated or batch start.
- Step 3 (anchor packets) is the one result that holds up: the EqF stays
  closest to 1 and FEJ stays overconfident whenever the survey error matters.
