import numpy as np
import numexpr as ne
from math import pi

from src.helpers import MultithreadedRNG
from src.surfaces import PM, ldis_deepwater
from src.surfaces import e_delta, directional_spectrum

class Surface:
    """"Generation of surface realizations from spectrum"""


    def __init__(self, xbounds, ybounds, kmax, surface_dict):
        """
        Setup random generator used for realizations
        spectrum is a scalar, 1-D array or 2-D array
        """
        self.kmax = kmax
        self.dx = 2 * pi / kmax
        self.surface_dict = surface_dict

        # setup x and y axes
        self.xbounds = xbounds
        Nx = int((xbounds[1] - xbounds[0]) / self.dx + 2)

        if Nx % 2: Nx += 1
        x_i = np.arange(Nx)
        self.x_a = x_i * self.dx + xbounds[0] - self.dx

        self.ybounds = ybounds
        if ybounds is not None:
            Ny = int((ybounds[1] - ybounds[0]) / self.dx + 2)
            if Ny % 2: Ny += 1
            y_i = np.arange(Ny)
            self.y_a = y_i * self.dx + ybounds[0] - self.dx
        else:
            self.y_a = None

        # wavenumber and spectrum variables set according to surface type
        self.kx = None
        self.ky = None
        self.spec_1D = None
        self.spec_2D = None
        self.surface_type = None
        self.omega = None
        self.seed = None
        self._surface_from_dict(surface_dict)

        # setup rng
        #self.rng = MultithreadedRNG(2 * self.N, seed=seed)
        self.rng = np.random.default_rng(self.seed)


    def _surface_from_dict(self, surface_dict):
        """deal with flat and sine special cases or generate a spectrum"""
        s_t = surface_dict['type']
        self.surface_type = s_t

        if 'seed' in s_t:
            self.seed = s_t['seed']
        else:
            self.seed = 0

        if s_t == 'sine':
            k = 2 * pi / surface_dict['L']
            theta = surface_dict['theta'] if 'theta' in surface_dict else 0.
            theta = np.deg2rad(theta)
            self.kx = np.array(k * np.cos(theta), ndmin=1)
            self.ky = np.array(k * np.sin(theta), ndmin=1)
            self.spec_1D = surface_dict['H'] / np.sqrt(8)
        elif s_t == 'flat':
            k = 0.
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
                k = ne.evaluate("sqrt(kx ** 2 + ky ** 2)")
                k_bearing = ne.evaluate("arctan2(ky, kx)")
            else:
                k = self.kx

        self.omega = ldis_deepwater(k)
        if s_t in ['sine', 'flat']:
            return

        # spectrum specifications

        if s_t == 'PM':
            self.spec_1D = PM(k, surface_dict['U20'])
        else:
            raise(ValueError("spectrum is not implimented"))

        if self.y_a is not None:
            delta = e_delta(k, surface_dict["U20"])
            self.spec_2D = directional_spectrum(delta,
                                                k,
                                                k_bearing,
                                                self.spec_1D)


    def realization(self):
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
            phase = self.omega * time
            if self.y_a is None:
                surf = np.sin(self.x_a[:, None] * self.kx[None, :] + phase)
            else:
                surf =  np.sin(self.x_a[:, None, None] * self.kx[None, None, :]
                               + self.y_a[None, :, None] * self.ky[None, None, :]
                               + phase)
            surf = surf.sum(axis=-1)
            return surf

        if time is not None:
            omega = self.omega
            phase = "exp(-1j * omega * time)"
        else:
            phase = "1."

        if realization is None:
            raise(ValueError("No surface specified"))


        # 1-D wave field
        if self.spec_2D is None:
            if derivative:
                kx = self.kx
                phase += " * -1j * kx"
            phase += " * realization"
            surface = np.fft.irfft(ne.evaluate(phase))

        # 2-D wave field
        else:
            if derivative is not None:
                if derivative == 'x':
                    kx = self.kx[:, None]
                    phase += " * -1j * kx "
                elif derivative == 'y':
                    ky = self.ky[None, :]
                    phase += " * -1j * ky"
                else:
                    raise(ValueError('Derivative must be either x or y'))
            phase += " * realization"
            spec = ne.evaluate(phase)
            surface = np.fft.irfft2(np.fft.ifftshift(spec, axes=1),
                                    axes=(1,0))
        return surface
