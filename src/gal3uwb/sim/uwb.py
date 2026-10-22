"""Listen-only UWB: anchors broadcast, the tag timestamps receptions on its own clock."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import C

_LIGHT_TIME_ITERS = 3  # fixed-point iterations; each shrinks the error by ~v/c


@dataclass(frozen=True)
class UwbData:
    T_rx: np.ndarray  # tag-clock reception timestamps, with noise [s]
    anchor: np.ndarray  # anchor index per reception
    t_tx: np.ndarray  # transmit timestamps as reported by the anchors [s]
    t_rx_true: np.ndarray  # true reception times, for diagnostics [s]


def box_anchors(half_x=3.0, half_y=3.0, z_low=0.2, z_high=2.8):
    """Eight anchors at the corners of a room-sized box."""
    xs, ys, zs = (-half_x, half_x), (-half_y, half_y), (z_low, z_high)
    return np.array([[x, y, z] for z in zs for x in xs for y in ys])


def simulate_uwb(traj, clock, anchors, rate, duration, sigma_range, rng, anchor_offsets=None, anchor_skews=None):
    """Round-robin broadcasts at `rate` Hz aggregate, in true time from t = 0.

    Anchor clocks are synchronized (to true time) unless offsets/skews are given;
    they only change the transmit timestamps the anchors report.
    """
    n_anchors = len(anchors)
    t_tx = np.arange(int(duration * rate)) / rate
    anchor = np.arange(len(t_tx)) % n_anchors
    a = anchors[anchor]

    t_rx = t_tx + np.linalg.norm(traj.position(t_tx) - a, axis=-1) / C
    for _ in range(_LIGHT_TIME_ITERS):
        t_rx = t_tx + np.linalg.norm(traj.position(t_rx) - a, axis=-1) / C

    keep = clock.covers(t_rx)  # drop receptions before the tag clock starts
    t_tx, anchor, t_rx = t_tx[keep], anchor[keep], t_rx[keep]
    T_rx = clock.tag_time(t_rx) + rng.normal(0.0, sigma_range / C, len(t_rx))
    offsets = np.zeros(n_anchors) if anchor_offsets is None else np.asarray(anchor_offsets)
    skews = np.zeros(n_anchors) if anchor_skews is None else np.asarray(anchor_skews)
    t_tx_reported = t_tx + offsets[anchor] + skews[anchor] * t_tx
    return UwbData(T_rx, anchor, t_tx_reported, t_rx)
