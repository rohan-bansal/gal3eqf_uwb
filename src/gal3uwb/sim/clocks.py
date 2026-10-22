"""Device clock models.

Convention: tag time T, true time t, tag offset tau = T - t, skew = dtau/dT.
With synchronized anchors the network runs on true time, and the filters
estimate delta = t - T = -tau and the time-rate bias b_rho = skew.

The tag clock is simulated on a uniform tag-time grid (the IMU samples on it).
An anchor clock reads A = t + o(t) and is simulated on a uniform true-time grid.
Both are piecewise linear between grid points, so forward and inverse maps are
exact inverses.
"""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import ClockNoise


@dataclass(frozen=True)
class TagClock:
    T: np.ndarray  # tag-time grid [s]
    tau: np.ndarray  # offset at each grid point [s]
    skew: np.ndarray  # skew state at each grid point [-]

    @classmethod
    def simulate(cls, n, dT, offset0, skew0, noise: ClockNoise, rng):
        T = np.arange(n) * dT
        w_tau = rng.normal(0.0, np.sqrt(noise.q_offset * dT), n - 1)
        w_skew = rng.normal(0.0, np.sqrt(noise.q_skew * dT), n - 1)
        skew = skew0 + np.concatenate([[0.0], np.cumsum(w_skew)])
        tau = offset0 + np.concatenate([[0.0], np.cumsum(skew[:-1] * dT + w_tau)])
        return cls(T, tau, skew)

    @property
    def t(self):
        return self.T - self.tau

    def covers(self, t):
        """Whether true times t lie within the simulated span."""
        return (t >= self.t[0]) & (t <= self.t[-1])

    def offset(self, T):
        """tau at tag times T."""
        _check_span(T, self.T)
        return np.interp(T, self.T, self.tau)

    def skew_at(self, T):
        _check_span(T, self.T)
        return np.interp(T, self.T, self.skew)

    def true_time(self, T):
        _check_span(T, self.T)
        return np.interp(T, self.T, self.t)

    def tag_time(self, t):
        _check_span(t, self.t)
        return np.interp(t, self.t, self.T)


@dataclass(frozen=True)
class AnchorClock:
    t: np.ndarray  # true-time grid [s]
    offset: np.ndarray  # o = A - t at each grid point [s]
    skew: np.ndarray  # do/dt at each grid point [-]

    @classmethod
    def simulate(cls, t0, n, dt, offset0, skew0, noise: ClockNoise, rng):
        t = t0 + np.arange(n) * dt
        w_off = rng.normal(0.0, np.sqrt(noise.q_offset * dt), n - 1)
        w_skew = rng.normal(0.0, np.sqrt(noise.q_skew * dt), n - 1)
        skew = skew0 + np.concatenate([[0.0], np.cumsum(w_skew)])
        offset = offset0 + np.concatenate([[0.0], np.cumsum(skew[:-1] * dt + w_off)])
        return cls(t, offset, skew)

    def offset_at(self, t):
        _check_span(t, self.t)
        return np.interp(t, self.t, self.offset)

    def skew_at(self, t):
        _check_span(t, self.t)
        return np.interp(t, self.t, self.skew)

    def true_time(self, A):
        """True time at which this anchor's clock reads A."""
        reading = self.t + self.offset
        _check_span(A, reading)
        return np.interp(A, reading, self.t)


def _check_span(x, grid):
    # np.interp clamps outside the grid, which would silently corrupt timestamps.
    if np.min(x) < grid[0] or np.max(x) > grid[-1]:
        raise ValueError(f"time outside simulated clock span [{grid[0]}, {grid[-1]}]")
