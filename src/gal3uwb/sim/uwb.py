"""Tag-transmit UWB, with synchronized or free-running anchors.

The tag transmits on a schedule in its own clock. Every anchor timestamps the
arrival in its own clock A_i. Anchors are static, so the arrival is exactly
t_tx + range / C. Measurements are stored in metres relative to the known
transmit time:
    y_i = C (A_i,rx - T_tx) = C (delta + Delta_i) + |p_tag - a_i| + noise,
where delta = A_0 - T (anchor 0, the network reference, minus tag time) and
Delta_i = A_i - A_0 (zero for synchronized anchors, which run on true time).

Free-running anchors also exchange packets: anchor i transmits on its own clock
and anchor j timestamps the arrival, giving
    y_ij = C (A_j,rx - A_i,tx) = C (Delta_j - Delta_i) + |a_j - a_i| + noise.
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


@dataclass(frozen=True)
class SyncData:
    """Anchor-to-anchor packets. The filters process each at its tag-time instant;
    a real filter would place it with its clock estimates, good to nanoseconds."""

    T: np.ndarray  # (M,) transmit times in tag time [s]
    tx: np.ndarray  # (M,) transmitting anchor
    y: np.ndarray  # (M, N) C (A_j,rx - A_i,tx) per receiving anchor j; NaN for j = tx [m]
    sigma: float  # [m]


def box_anchors(half_width=3.0, z_low=0.2, z_high=2.8):
    """Eight anchors on the corners of a box around the flight volume."""
    s = half_width
    return np.array([[x, y, z] for z in (z_low, z_high) for x in (-s, s) for y in (-s, s)])


def simulate_uwb(traj, clock, T_tx, anchors, lever_arm, sigma, rng, anchor_clocks=None):
    """Arrival timestamps at every anchor for each tag transmission at tag times T_tx.
    Without anchor_clocks the anchors are synchronized to true time."""
    tau = clock.offset(T_tx)
    t_tx = T_tx - tau
    p_tag = traj.position(t_tx) + traj.rotation(t_tx) @ lever_arm
    ranges = np.linalg.norm(p_tag[:, None, :] - anchors[None, :, :], axis=-1)
    if anchor_clocks is None:
        offsets = np.zeros_like(ranges)
    else:
        t_rx = t_tx[:, None] + ranges / C
        offsets = np.stack([c.offset_at(t_rx[:, i]) for i, c in enumerate(anchor_clocks)], -1)
    # Computed in metres so the large offsets do not cost precision.
    y = C * (offsets - tau[:, None]) + ranges + rng.normal(0.0, sigma, ranges.shape)
    return UwbData(T_tx, y, anchors, sigma)


def simulate_sync(tag_clock, anchor_clocks, anchors, rate, T_span, sigma, rng):
    """Each anchor broadcasts at `rate` on its own clock; all others timestamp it."""
    n = len(anchors)
    dist = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=-1)
    T_all, tx_all, y_all = [], [], []
    for i, clock in enumerate(anchor_clocks):
        # Readings of anchor i's clock spanning the tag-time window (offsets are under a few ms).
        t_lo, t_hi = tag_clock.true_time(np.array(T_span))
        A_lo = t_lo + clock.offset_at(t_lo) + rng.uniform(0.0, 1.0 / rate)
        A_tx = np.arange(A_lo, t_hi + clock.offset_at(t_hi), 1.0 / rate)
        t_tx = clock.true_time(A_tx)
        t_tx = t_tx[tag_clock.covers(t_tx)]
        T_e = tag_clock.tag_time(t_tx)
        keep = (T_e >= T_span[0]) & (T_e < T_span[1])
        t_tx, T_e = t_tx[keep], T_e[keep]
        y = np.full((len(t_tx), n), np.nan)
        for j, other in enumerate(anchor_clocks):
            if j == i:
                continue
            t_rx = t_tx + dist[i, j] / C
            noise = rng.normal(0.0, sigma, len(t_tx))
            y[:, j] = dist[i, j] + C * (other.offset_at(t_rx) - clock.offset_at(t_tx)) + noise
        T_all.append(T_e)
        tx_all.append(np.full(len(T_e), i))
        y_all.append(y)
    T, tx, y = np.concatenate(T_all), np.concatenate(tx_all), np.concatenate(y_all)
    order = np.argsort(T)
    return SyncData(T[order], tx[order], y[order], sigma)
