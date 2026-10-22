"""Analytic drone trajectories, evaluated in true time (vectorized)."""

from dataclasses import dataclass

import numpy as np

from gal3uwb.params import GRAVITY

_FD_STEP = 1e-4  # central-difference step for angular velocity [s]


@dataclass(frozen=True)
class CircleTrajectory:
    """Horizontal circle at constant speed; speed 0 is a hover at (radius, 0, height).

    Attitude is that of a quadrotor: body z along the thrust (a - g), fixed yaw.
    """

    speed: float
    radius: float = 2.0
    height: float = 1.0
    yaw: float = 0.0

    @property
    def rate(self):
        return self.speed / self.radius

    def position(self, t):
        t = np.asarray(t, dtype=float)
        wt = self.rate * t
        return np.stack([self.radius * np.cos(wt), self.radius * np.sin(wt), np.full_like(t, self.height)], -1)

    def velocity(self, t):
        t = np.asarray(t, dtype=float)
        wt = self.rate * t
        return self.speed * np.stack([-np.sin(wt), np.cos(wt), np.zeros_like(t)], -1)

    def acceleration(self, t):
        t = np.asarray(t, dtype=float)
        wt = self.rate * t
        return -self.speed * self.rate * np.stack([np.cos(wt), np.sin(wt), np.zeros_like(t)], -1)

    def rotation(self, t):
        """World-from-body rotations, shape (..., 3, 3)."""
        z = self.acceleration(t) - GRAVITY
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        x_c = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        y = np.cross(z, x_c)
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        x = np.cross(y, z)
        return np.stack([x, y, z], -1)

    def angular_velocity(self, t):
        """Body-frame angular velocity, from [w]x = R^T dR/dt."""
        t = np.asarray(t, dtype=float)
        dR = (self.rotation(t + _FD_STEP) - self.rotation(t - _FD_STEP)) / (2 * _FD_STEP)
        W = np.einsum("...ji,...jk->...ik", self.rotation(t), dR)
        return 0.5 * np.stack([W[..., 2, 1] - W[..., 1, 2], W[..., 0, 2] - W[..., 2, 0], W[..., 1, 0] - W[..., 0, 1]], -1)

    def specific_force(self, t):
        """Body-frame specific force R^T (a - g)."""
        return np.einsum("...ji,...j->...i", self.rotation(t), self.acceleration(t) - GRAVITY)
