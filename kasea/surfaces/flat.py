import numpy as np
from kasea.surfaces.base import Surface


class Flat(Surface):
    """"Generation of surface realizations from spectrum"""

    def __init__(self, experiment):
        """
        Setup random generator used for realizations spectrum is a scalar, 1-D array or 2-D array
        """
        super().__init__(experiment)

        # flat surface does not need a z buffer
        self.est_z_max = 0.
        super().set_bounds(self.est_z_max)
        self.surface_type = 'flat'


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        if self.y_a is None:
            return np.zeros_like(self.x_a)
        else:
            return np.zeros((self.x_a.size, self.y_a.size), dtype=np.float64)
