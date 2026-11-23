"""EKF with first-estimate Jacobians (FEJ): the standard fix for EKF-SLAM inconsistency.

Why it is needed. Rotating the whole scene about gravity, or translating it,
changes no reading, so no filter should ever gain information along those
directions. In the EKF's coordinates the rotation is the direction

    N(x) = (R^T e_z, e_z x v, e_z x p, 0 (biases, clocks), e_z x a_0, ..., e_z x a_{N-1})

(translation is dp = da_i = t). It depends on the point x at which the
Jacobians are evaluated. A measurement Jacobian H(x) is blind to N(x), and a
transition Jacobian from x to x' carries N(x) to N(x'), so as long as all
Jacobians are evaluated along one unbroken chain of points the direction stays
unobservable. The plain EKF breaks the chain at every update: the covariance's
blind direction is N(x-), at the estimate before the update, but the next
transition starts from the corrected x+. Each update leaks a little information
into the scene's heading, and the filter becomes overconfident
(Huang, Mourikis & Roumeliotis, ISER 2008 / IJRR 2010).

FEJ evaluates every Jacobian at points that never jump:
  - anchors: their first estimate (the survey guess), for the whole flight;
  - drone: its propagated estimate, before any update at that time (x-).
The step out of an update then starts at x- and ends at the propagated estimate
x', in the closed form of Li & Mourikis (IJRR 2013), here for a body-frame
attitude error:

    F_theta,theta = R'^T R-
    F_v,theta     = -[v' - v- - g dT]x R-
    F_p,theta     = -[p' - p- - v- dT - g dT^2 / 2]x R-

which carries N(x-) exactly to N(x'). Between updates x- is the current estimate
and these are the plain EKF's Jacobians. The attitude reset after an update is
dropped: F_theta,theta already contains the rotation from x- to x+.

The price is Jacobians at stale points: with a 30 cm survey guess, every anchor
Jacobian stays that far off for the whole flight. The EqF (eqf_slam.py) gets
the same unobservable directions with Jacobians at the current estimate,
because in its coordinates N is a constant vector.
"""

from gal3uwb import lie
from gal3uwb.filters.common import POS, ROT, VEL
from gal3uwb.filters.ekf import propagate
from gal3uwb.params import GRAVITY

from ekf_slam import EkfSlam


class EkfFej(EkfSlam):
    name = "FEJ-EKF"
    reset_attitude = False

    def __init__(self, x0, P0, noise):
        super().__init__(x0, P0, noise)
        self.lin = x0.nav  # drone: last propagated estimate (x-)
        self.anchors0 = x0.anchors.copy()  # anchors: first estimate

    def transition(self, gyro, accel, dT):
        nav, F, F_input = propagate(self.x.nav, gyro, accel, dT)
        lin = self.lin
        F[ROT, ROT] = nav.R.T @ lin.R
        F[VEL, ROT] = -lie.hat(nav.v - lin.v - GRAVITY * dT) @ lin.R
        F[POS, ROT] = -lie.hat(nav.p - lin.p - lin.v * dT - 0.5 * GRAVITY * dT**2) @ lin.R
        return nav, F, F_input

    def propagate(self, gyro, accel, dT):
        super().propagate(gyro, accel, dT)
        self.lin = self.x.nav

    def jacobian_point(self):
        return self.lin, self.anchors0
