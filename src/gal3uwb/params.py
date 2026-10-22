"""Physical constants and noise parameters shared by the simulator and filters."""

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0  # speed of light [m/s]
GRAVITY = np.array([0.0, 0.0, -9.81])  # world-frame gravity [m/s^2]


@dataclass(frozen=True)
class ImuNoise:
    gyro_nd: float = 2.5e-4  # gyro white noise density [rad/s/sqrt(Hz)]
    accel_nd: float = 2.0e-3  # accel white noise density [m/s^2/sqrt(Hz)]
    gyro_bias_rw: float = 1.0e-5  # gyro bias random walk [rad/s^2/sqrt(Hz)]
    accel_bias_rw: float = 1.0e-4  # accel bias random walk [m/s^3/sqrt(Hz)]


@dataclass(frozen=True)
class ClockNoise:
    """Tag clock noise, defined in tag time T.

    The offset is tau = T - t (tag time minus true time) and the skew is dtau/dT.
    Defaults are TCXO-like (h0 = 2e-19, h-2 = 2e-20).
    """

    q_offset: float = 1.0e-19  # white frequency noise PSD on the offset [s^2/s]
    q_skew: float = 3.9e-19  # random walk PSD on the skew [1/s]
