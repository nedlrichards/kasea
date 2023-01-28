import numpy as np
import numexpr as ne
from math import pi

from scipy.interpolate import interp1d

from kasea.surfaces import ldis_deepwater
from kasea.surfaces import e_delta, directional_spectrum
from kasea.surfaces.base import Surface

g = 9.81

class Spectrum(Surface):
    """general spectrum"""

    def __init__(self, experiment, est_z_max):
        """
        Setup random generator used for realizations spectrum is a scalar, 1-D array or 2-D array
        """
        super().__init__(experiment)
        self.est_z_max = est_z_max
        super().set_bounds(self.est_z_max)

        # compute wavenumber spectrum from x_axis
        dx = (self.x_a[-1] - self.x_a[0]) / (self.x_a.size - 1)
        self.ks = 2 * np.pi / dx
        # positive only axis for x
        self.kx = self.ks * np.arange(self.x_a.size // 2 + 1) / self.x_a.size
        # positive/negative axis for y
        self.ky = self.ks * (np.arange(self.y_a.size) - self.y_a.size // 2)
        self.ky /= self.y_a.size


    def compute_specta(self, *args):
        """Compute both 1D and 2D spectrum"""
        # compute 1D and 2D spectra
        self.spec_1D = self.compute_1D(self.kx, *args)

        kx = self.kx[:, None]
        ky = self.ky[None, :]
        k = ne.evaluate("sqrt(kx ** 2 + ky ** 2)")
        k_bearing = ne.evaluate("arctan2(ky, kx)")

        self.omega = ldis_deepwater(k)
        self.delta = e_delta(k, self.U20)
        omni_ier = interp1d(self.kx, self.spec_1D, bounds_error=False, fill_value=0.)
        omni_spectrum = omni_ier(k)

        self.spec_2D = directional_spectrum(self.delta, k, k_bearing, omni_spectrum)


    def gen_realization(self):
        """Generate a realization of the surface spectrum"""
        N = self.kx.size * self.ky.size

        samps = self.rng.normal(size=2 * N)
        samps = samps.view(np.complex128)

        # 2-D wave field
        samps = samps.reshape(self.spec_2D.shape)
        samps[0, :] = np.sqrt(2) * np.real(samps[0, :])
        samps[-1, :] = np.sqrt(2) * np.real(samps[-1, :])

        spec_2D = self.spec_2D
        kmax = self.kmax
        expr = "sqrt(spec_2D * N * kmax ** 2) * samps"
        spec = ne.evaluate(expr)
        return spec


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""

        if time is not None:
            # use deep-water dispersion to compute frequency axis
            omega = self.omega
            phase = "exp(-1j * omega * time)"
        else:
            phase = "1."

        if realization is None:
            raise(ValueError("No surface specified"))

        kx = self.kx[:, None]
        ky = self.ky[None, :]
        if derivative == 'x':
            phase += " * 1j * kx "
        elif derivative == 'y':
            phase += " * 1j * ky"
        elif derivative == 'xx':
            phase += " -1 * kx ** 2"
        elif derivative == 'xy':
            phase += " -1 * kx * ky"
        elif derivative == 'yy':
            phase += " -1 * ky ** 2"
        elif derivative is not None:
            raise(ValueError('Derivative must be either x, y, xx, xy, yy'))

        phase += " * realization"
        spec = ne.evaluate(phase)
        surface = np.fft.irfft2(np.fft.ifftshift(spec, axes=1), axes=(1,0))

        return surface


    def compute_1D(self, kx, *args):
        """specification of one dimensional spectrum"""
        pass
