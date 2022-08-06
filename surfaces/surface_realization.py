import numpy as np
import numexpr as ne
from math import pi
from scipy.interpolate import RectBivariateSpline

from src.surfaces import PM, ldis_deepwater
from src.surfaces import e_delta, directional_spectrum
from src import bound_axes

class Surface:
    """"Generation of surface realizations from spectrum"""

    def __init__(self, experiment):
        """
        Setup random generator used for realizations
        spectrum is a scalar, 1-D array or 2-D array
        """

        self.dr = experiment.dr
        self.z_src = experiment.z_src
        self.z_rcr = experiment.z_rcr

        self.kmax = 2 * pi / experiment.dx
        self.dx = experiment.dx

        self.surface_dict = experiment.toml_dict['surface']

        self.theta = self.surface_dict['theta']
        self.num_snaps = self.surface_dict['num_snaps']

        bounds = bound_axes(self.z_src, self.z_rcr, self.dr,
                            experiment.est_z_max, experiment.max_dur,
                            c=experiment.c, theta=self.theta)

        # setup x and y axes
        self.xbounds = bounds[0]
        n_start = self.xbounds[0] // self.dx
        Nx = int((self.xbounds[1] - self.xbounds[0]) / self.dx + 1)
        if Nx % 2: Nx += 1
        self.x_a = (np.arange(Nx) + n_start - 1) * self.dx

        self.ybounds = bounds[1]
        if self.ybounds is not None:
            n_start = self.ybounds[0] // self.dx
            Ny = int((self.ybounds[1] - self.ybounds[0]) / self.dx)
            if Ny % 2: Ny += 1
            self.y_a = (np.arange(Ny) + n_start - 1) * self.dx
        else:
            self.y_a = None

        # wavenumber and spectrum variables set according to surface type
        self.kx = None
        self.ky = None
        self.spec_1D = None
        self.spec_2D = None
        self.theta = None
        self.surface_type = None
        self.omega = None
        self.seed = None
        self._surface_from_dict(self.surface_dict)

        # setup rng
        self.rng = np.random.default_rng(self.seed)


    def _surface_from_dict(self, sd):
        """deal with flat and sine special cases or generate a spectrum"""
        s_t = sd['type']
        self.surface_type = s_t
        self.seed = sd['seed'] if 'seed' in sd else 0
        theta = sd['theta'] if 'theta' in sd else 0.
        self.theta = theta

        if s_t == 'sine':
            self.k = np.array(2 * pi / sd['L'], ndmin=1)
            theta = np.deg2rad(theta)
            self.spec_1D = sd['H'] / np.sqrt(8)
        elif s_t == 'flat':
            self.k = 0.
            self.kx = 0.
            self.spec_1D = 0.
        else:
            # spectrum specifications
            Nx = self.x_a.size
            self.kx = np.arange(Nx / 2 + 1) * self.kmax / Nx

            if self.y_a is not None:
                Ny = self.y_a.size
                self.ky = (np.arange(Ny) - Ny // 2) * self.kmax / Ny

                kx = self.kx[:, None]
                ky = self.ky[None, :]
                self.k = ne.evaluate("sqrt(kx ** 2 + ky ** 2)")
                k_bearing = ne.evaluate("arctan2(ky, kx)")
            else:
                self.k = self.kx

        self.omega = ldis_deepwater(self.k)
        if s_t in ['sine', 'flat']:
            return

        # spectrum specifications
        if s_t == 'PM':
            self.spec_1D = PM(self.k, sd['U20'])
        else:
            raise(ValueError("spectrum is not implimented"))

        if self.y_a is not None:
            delta = e_delta(self.k, sd["U20"])
            self.spec_2D = directional_spectrum(delta,
                                                self.k,
                                                k_bearing,
                                                self.spec_1D)


    def gen_realization(self):
        """Generate a realization of the surface spectrum"""
        s_t = self.surface_dict['type']
        if s_t in ['flat', 'sine']:
            return

        N = self.kx.size if self.y_a is None else self.kx.size * self.ky.size

        samps = self.rng.normal(size=2 * N)
        #self.rng.fill()
        #samps = self.rng.values.view(np.complex128)
        samps = samps.view(np.complex128)

        # 1-D wave field
        if self.spec_2D is None:
            # inverse scaling of Heinzel et al. (2002), Eq 24
            abs_k2 = self.spec_1D * self.kx.size * self.kmax / 2
            return samps * np.sqrt(abs_k2)

        # 2-D wave field
        samps = samps.reshape(self.spec_2D.shape)
        #abs_k2 = self.spec_2D * self.Nx * self.Ny * self.kmax ** 2
        #realization = np.sqrt(abs_k2) * samps
        spec_2D = self.spec_2D
        Nx = self.kx.size
        Ny = self.ky.size
        kmax = self.kmax
        expr = "sqrt(spec_2D * Nx * Ny * kmax ** 2) * (samps / sqrt(2))"
        return ne.evaluate(expr)


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""

        s_t = self.surface_dict['type']

        if s_t == 'flat':
            if self.y_a is None:
                return np.zeros(self.x_a.size, dtype=np.float64)
            else:
                return np.zeros((self.x_a.size, self.y_a.size),
                                dtype=np.float64)

        if s_t == 'sine':
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

        if time is not None:
            omega = self.omega
            phase = "exp(-1j * omega * time)"
        else:
            phase = "1."

        # 1-D wave field
        if self.spec_2D is None:
            if derivative:
                kx = self.kx
                phase += " * 1j * kx"
            phase += " * realization"
            surface = np.fft.irfft(ne.evaluate(phase))

        # 2-D wave field
        else:
            if derivative is not None:
                if derivative == 'x':
                    kx = self.kx[:, None]
                    phase += " * 1j * kx "
                elif derivative == 'y':
                    ky = self.ky[None, :]
                    phase += " * 1j * ky"
                else:
                    raise(ValueError('Derivative must be either x or y'))
            phase += " * realization"
            spec = ne.evaluate(phase)
            surface = np.fft.irfft2(np.fft.ifftshift(spec, axes=1),
                                    axes=(1,0))

        return surface
