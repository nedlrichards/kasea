import numpy as np
import numexpr as ne
from math import pi

from scipy.interpolate import interp1d

from kasea.surfaces import ldis_deepwater
from kasea.surfaces import e_delta, directional_spectrum
from kasea.surfaces.spectrum import Spectrum

g = 9.81

class Gaussian(Spectrum):
    """Gaussian roughness spectrum"""


    def __init__(self, experiment):
        """
        Setup random generator used for realizations spectrum is a scalar, 1-D array or 2-D array
        """
        self.l_corr = experiment.toml_dict['surface']['corr_length']
        self.h_rms = experiment.toml_dict['surface']['rms_height']
        self.U20 = experiment.toml_dict['surface']['U20']
        # add buffer equal to 2 x RMS wave height
        self.est_z_max = 2 * self.h_rms
        super().__init__(experiment, self.est_z_max)
        self.surface_type = 'Gaussian'
        self.compute_specta(self.l_corr, self.h_rms)


    def compute_1D(self, kx, l, h):
        """Compute the Gaussian spectrum from wavenumber"""

        k = np.array(kx, ndmin=1)
        expr = "2 * l * h ** 2 / (2 * sqrt(pi)) * exp(-k ** 2 * l ** 2 / 4)"
        spec = ne.evaluate(expr)
        spec[np.isnan(spec)] = 0

        return spec
