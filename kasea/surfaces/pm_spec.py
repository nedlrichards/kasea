import numpy as np
import numexpr as ne
from math import pi

from scipy.interpolate import interp1d

from kasea.surfaces import ldis_deepwater
from kasea.surfaces import e_delta, directional_spectrum
from kasea.surfaces.spectrum import Spectrum

g = 9.81

class PM(Spectrum):
    """Pierson-Moskovitz spectrum"""


    def __init__(self, experiment):
        """
        Setup random generator used for realizations spectrum is a scalar, 1-D array or 2-D array
        """
        self.U20 = experiment.toml_dict['surface']['U20']
        # add buffer equal to 2 x significant wave height
        self.est_z_max = 0.22 * self.U20 ** 2 / g
        self.h_rms = 2.74e-3 * self.U20 ** 4 / g ** 2
        super().__init__(experiment, self.est_z_max)
        self.surface_type = 'PM'
        self.compute_specta(self.U20)


    def compute_1D(self, axis_in, U20, is_omega=False):
        """Compute the Pierson-Moskowitz spectrum by radial frequency"""

        if is_omega:
            omega = np.array(axis_in, ndmin=1)
        else:
            k = np.array(axis_in, ndmin=1)
            # use deep-water dispersion to compute frequency axis
            omega = ldis_deepwater(k)
            dwdk = ldis_deepwater(k, derivative=True)

        # Pierson Moskowitz parameters
        beta = 0.74
        alpha = 8.1e-3

        expr = "exp(-beta * g ** 4 / (omega * U20) ** 4) * alpha * g ** 2 / omega ** 5"
        if not is_omega:
            expr = 'dwdk * ' + expr

        pm = ne.evaluate(expr)

        pm[omega <= 0] = 0
        pm[np.isnan(pm)] = 0

        return pm
