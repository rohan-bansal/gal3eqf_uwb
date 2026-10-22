"""Tag-transmit UWB with synchronized anchors.

The tag transmits on a schedule in its own clock. Every anchor timestamps the
arrival in network time, which synchronized anchors share and which we take as
true time. Anchors are static, so the arrival is exactly t_tx + range / C.

Measurements are stored in metres relative to the known transmit time:
    y_i = C (t_rx,i - T_tx) = C delta + |p_tag - a_i| + noise,
where delta = t - T (network minus tag time) at the transmission.
"""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import C


@dataclass(frozen=True)
class UwbData:
    T_tx: np.ndarray  # (J,) transmit times in tag time [s]
    y: np.ndarray  # (J, N) C (t_rx - T_tx) per anchor [m]
    anchors: np.ndarray  # (N, 3) [m]
    sigma: float  # timestamp noise in metres [m]


def box_anchors(half_width=3.0, z_low=0.2, z_high=2.8):
    """Eight anchors on the corners of a box around the flight volume."""
    s = half_width
    return np.array([[x, y, z] for z in (z_low, z_high) for x in (-s, s) for y in (-s, s)])


def simulate_uwb(traj, clock, T_tx, anchors, lever_arm, sigma, rng):
    """Arrival times at every anchor for each tag transmission at tag times T_tx."""
    delta = -clock.offset(T_tx)
    t_tx = T_tx + delta
    p_tag = traj.position(t_tx) + traj.rotation(t_tx) @ lever_arm
    ranges = np.linalg.norm(p_tag[:, None, :] - anchors[None, :, :], axis=-1)
    y = C * delta[:, None] + ranges + rng.normal(0.0, sigma, ranges.shape)
    return UwbData(T_tx, y, anchors, sigma)
