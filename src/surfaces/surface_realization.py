import numpy as np
import numexpr as ne
from math import pi

from src.helpers import MultithreadedRNG
from src.surfaces import PM
from src.surfaces import e_delta, directional_spectrum

class Surface:
    """"Generation of surface realizations from spectrum"""


    def __init__(self, xbounds, ybounds, kmax, U20, seed=None):
        """Setup random generator used for realizations
        spectrum is a scalar, 1-D array or 2-D array
        """
        self.rng = np.random.default_rng(seed=seed)
        self.kmax = kmax
        self.dx = 2 * pi / kmax

        self.xbounds = xbounds
        Nx = int((xbounds[1] - xbounds[0]) / self.dx + 2)
        if Nx % 2: Nx += 1
        x_i = np.arange(Nx)
        self.x_a = x_i * self.dx + xbounds[0] - self.dx
        # kx is the real fft axis
        self.Nx = Nx // 2 + 1
        self.kx = np.arange(self.Nx) * kmax / Nx

        self.ybounds = ybounds
        if ybounds is not None:
            self.Ny = int((ybounds[1] - ybounds[0]) / self.dx + 2)
            if self.Ny % 2: self.Ny += 1
            y_i = np.arange(self.Ny)
            self.y_a = y_i * self.dx + ybounds[0] - self.dx
            # kx is a complex fft axis
            self.ky = (np.arange(self.Ny) - self.Ny // 2) * kmax / self.Ny
            self.N = self.Nx * self.Ny
        else:
            self.N = self.Nx
            self.y_a = None
            self.ky = None
            self.Ny = None

        self.g = 9.81
        self.km = 370  # wavenumber at GC wave phase speed minimum

        if ybounds is not None:
            kx = self.kx[:, None]
            ky = self.ky[None, :]

            #k = np.sqrt(self.kx[:, None] ** 2 + self.ky[None, :] ** 2)
            k = ne.evaluate("sqrt(kx ** 2 + ky ** 2)")
            self.spec_1D = PM(k, U20)
            k_bearing = ne.evaluate("arctan2(ky, kx)")
            delta = e_delta(k, U20)
            self.spec_2D = directional_spectrum(delta, k, k_bearing, self.spec_1D)
        else:
            self.spec_1D = PM(self.kx, U20)
            self.spec_2D = None


    def realization(self):
        """Generate a realization of the surface spectrum"""
        #samps = self.rng.normal(size=2 * self.N, scale=)
        rng = MultithreadedRNG(2 * self.N)
        rng.fill()
        samps = rng.values.view(np.complex128)

        # 1-D wave field
        if self.spec_2D is None:
            # inverse scaling of Heinzel et al. (2002), Eq 24
            abs_k2 = self.spec_1D * self.N * self.kmax / 2
            return samps * np.sqrt(abs_k2)

        # 2-D wave field
        samps = samps.reshape(self.spec_2D.shape)
        #abs_k2 = self.spec_2D * self.Nx * self.Ny * self.kmax ** 2
        #realization = np.sqrt(abs_k2) * samps
        spec_2D = self.spec_2D
        Nx = self.Nx
        Ny = self.Ny
        kmax = self.kmax
        expr = "sqrt(spec_2D * Nx * Ny * kmax ** 2) * (samps / sqrt(2))"
        return ne.evaluate(expr)


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        if time is not None:
            omega = self.omega
            phase = "exp(-1j * omega * time)"
        else:
            phase = "1."

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


    def ldis_deepwater(self, wave_number):
        """linear dispersion relationship assuming deep water"""
        gc = (1 + (wave_number / self.km) ** 2)
        omega = np.sqrt(self.g * wave_number * gc)
        return omega
