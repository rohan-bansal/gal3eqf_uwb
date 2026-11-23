# 00: EqF on Gal(3) vs EKF, surveyed anchors

The EqF on Gal(3) x| gal(3) (Delama et al.) against the error-state EKF, with
known (surveyed) anchor positions, for three clock scenarios:

- step 1 (`ideal_clock`): synchronized anchors, unknown constant tag offset;
- step 2 (`gaussian_skew`): synchronized anchors, tag skew ~ N(0, 20 ppm^2) and TCXO noise;
- step 3 (`unsynced`): a TCXO clock per anchor, anchor-to-anchor packets at 10 Hz per anchor.

The filters live in `src/gal3uwb/filters/` (`ekf.py`, `eqf.py`).

| Script | What it runs | Results |
|---|---|---|
| `synthetic.py` | circle flights at 0, 0.5, 2, 5 m/s, 50 per setting | `results/synthetic_steps12`, `results/synthetic_step3` |
| `synthetic.py --speeds 5 10 15 25 35 45 --turn-accel 12.5` | 5 to 45 m/s with the scene scaled by v^2 | `results/synthetic_highspeed` |
| `synthetic.py --pos-sigma 0.3 --gyro-bias-sigma 0.01` | pessimistic start | `results/synthetic_steps12_pessimistic` |
| `miluv.py` | ten MILUV flights, real path and IMU, simulated UWB | `results/miluv_steps12`, `results/miluv_step3` |
| `eqf_tuning.py` | prior on the EqF's virtual bias b_nu | `results/eqf_tuning` |

Run from the repo root through Slurm, for example

    sbatch slurm/experiment.sbatch experiments/00_eqf_vs_ekf/synthetic.py --scenarios unsynced --out results/synthetic_step3

Finding: equal accuracy everywhere; the EqF is more consistent only near hover
(mostly in heading), and the Galilean clock coupling is negligible at every
speed up to 45 m/s.
