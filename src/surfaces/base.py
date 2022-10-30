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

        self.max_dur = experiment.max_dur
        self.c = experiment.c

        # wavenumber and spectrum variables set according to surface type
        self.tau_max = experiment.tau_max
        self.xbounds = None
        self.x_a = None
        self.ybounds = None
        self.y_a = None

        self.kx = None
        self.ky = None
        self.spec_1D = None
        self.spec_2D = None
        self.theta = experiment.theta
        self.surface_type = None
        self.omega = None
        self.seed = experiment.seed

        self.est_z_max = None  # set by child surface class

        # setup rng
        self.rng = np.random.default_rng(self.seed)


    def set_bounds(self, est_z_max):
        """The estimate of z max will be provided by the child class"""
        self.est_z_max = est_z_max
        bounds = bound_axes(self.z_src, self.z_rcr, self.dr,
                            est_z_max, self.max_dur,
                            c=self.c, theta=self.theta)

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

        # restrict x and y axis with accurate time bounds
        if self.y_a is None or (self.theta.size == 1 and np.abs(self.theta[0]) < 0.01):
            return

        r_src = np.sqrt(self.x_a[:, None] ** 2 + self.y_a[None, :] ** 2
                        + (est_z_max + self.z_src) ** 2)

        mask = np.zeros((self.x_a.size, self.y_a.size), dtype=np.bool_)

        for th in self.theta:
            r_rcr = np.sqrt((self.dr * np.cos(th) - self.x_a[:, None]) ** 2
                            + (self.dr * np.sin(th) - self.y_a[None, :]) ** 2
                            + (self.z_rcr + est_z_max) ** 2)

            tau_ras = (r_src + r_rcr) / self.c
            mask |= tau_ras < self.tau_max

        self.mask = mask

        self.x_a = self.x_a[np.any(mask, axis=1)]
        self.y_a = self.y_a[np.any(mask, axis=0)]


    def gen_realization(self):
        """Generate a realization of the surface spectrum"""
        pass


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        pass
