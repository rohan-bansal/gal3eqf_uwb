"""Device clock models.

Convention: tag time T, true (network) time t, offset tau = T - t, skew = dtau/dT.
The filters estimate delta = t - T = -tau and the time-rate bias b_rho = skew.
The clock is simulated on a uniform tag-time grid and is piecewise linear
between grid points, so the T <-> t maps are exact inverses.
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


def _check_span(x, grid):
    # np.interp clamps outside the grid, which would silently corrupt timestamps.
    if np.min(x) < grid[0] or np.max(x) > grid[-1]:
        raise ValueError(f"time outside simulated clock span [{grid[0]}, {grid[-1]}]")
