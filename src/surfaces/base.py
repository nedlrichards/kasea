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
        self.max_dur = experiment.max_dur
        self.c = experiment.c


        # wavenumber and spectrum variables set according to surface type
        self.xbounds = None
        self.x_a = None
        self.ybounds = None
        self.y_a = None

        self.kx = None
        self.ky = None
        self.spec_1D = None
        self.spec_2D = None
        theta = self.surface_dict['theta'] if 'theta' in self.surface_dict else 0.
        self.surface_type = None
        self.omega = None
        self.seed = self.surface_dict['seed'] if 'seed' in self.surface_dict else 0

        # setup rng
        self.rng = np.random.default_rng(self.seed)


    def set_bounds(self, est_z_max):
        """The estimate of z max will be provided by the child class"""
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


    def gen_realization(self):
        """Generate a realization of the surface spectrum"""
        pass


    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        pass
