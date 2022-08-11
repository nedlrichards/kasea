import numpy as np
import numexpr as ne
from math import pi
from scipy.interpolate import RectBivariateSpline

from src.surfaces import PM, ldis_deepwater
from src.surfaces import e_delta, directional_spectrum
from src import bound_axes

from ldis import ldis_deepwater

class Sine(Surface):
    """"Generation of surface realizations from spectrum"""

    def __init__(self, experiment):
        """
        Setup random generator used for realizations spectrum is a scalar, 1-D array or 2-D array
        """
        super().__init__(experiment)

        self.k = np.array(2 * pi / self.surface_dict['L'], ndmin=1)
        self.spec_1D = self.surface_dict['H'] / np.sqrt(8)

        self.omega = ldis_deepwater(self.k)


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        if time is None:
            time = 0

        if self.y_a is None:
            phase = self.x_a[:, None] * self.k[None, :] \
                    + self.omega[None, :] * time
        else:
            phase = self.x_a[:, None, None] * self.k[None, None, :] \
                    + self.y_a[None, :, None] * 0. \
                    + self.omega[None, None, :] * time

        spec_1D = self.spec_1D
        surf = ne.evaluate("spec_1D * exp(1j * phase) * sqrt(2)")
        if derivative == 'x':
            surf *= 1j * self.k[None :]
        elif derivative == 'y':
            surf *= 0.
        surf = surf.sum(axis=-1)
        return np.imag(surf)
