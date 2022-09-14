import numpy as np
import numexpr as ne

class Eta:
    """Specification of a single frozen surface realization"""

    def __init__(self, surface_position, gradient, hessian=None, x_a, y_a=None):
        """surface postion is a 2xM or a 3xMxN array for a 2 or 3D geometry"""
        self.surface_position = surface_position
        self.x_a = x_a
        if self.surface_position.shape[0] == 3 and y_a is None:
            raise(ValueError("2D surface requires a y axis"))
        self.y_a = y_a

    def distance_mask(self, z_src, r_rcr, d_max):
        """Return surface positions within a total range of d_max"""
        z = self.surface_position
        x_r = r_rcr[0]
        z_r = r_rcr[-1]
        if self.y_a is None:
            x_a = self.x_a
            y_a = 0
            y_r = 0
        else:
            x_a = self.x_a[:, None]
            y_a = self.y_a[None, :]
            y_r = r_rcr[1]

        d_src = ne.evaluate("sqrt(x_a ** 2 + y_a ** 2 + (z - z_src) ** 2)")
        d_rcr = ne.evaluate("sqrt((x_r - x_a) ** 2 + (y_r - y_a) ** 2 + (z_r - z) ** 2)")
        return (d_src + d_rcr) <= d_max

    def shadow_mask(self, z_src, r_rcr, distance_mask):
        """Create mask where either source or receiver ray is shadowed"""
        pass

    def complete_integral(self, z_src, r_rcr, d_max):
