import tomli
import numpy as np
import numexpr as ne
from math import pi
import sys
import importlib.util
import os

from src.surfaces import Surface
from scipy.interpolate import RectBivariateSpline
from src.helpers import bound_axes

class SurfGen:
    """prepare surfaces specified by a test scenario"""


    def __init__(self, broadcast):
        """scatter calculation specification load and basic setup"""
        self.broadcast = broadcast
        self.dx = broadcast.dx
        kmax = 2 * pi / self.dx
        self.theta = broadcast.toml_dict['surface']['theta']

        inner_bounds =  bound_axes(broadcast.src, broadcast.rcr, broadcast.dx,
                                   broadcast.est_z_max, broadcast.max_dur,
                                   c=broadcast.c, to_rotate=False)

        if 'theta' in broadcast.toml_dict['surface'] \
                and np.any(np.abs(self.theta)) > 1e-5:
            self.to_rotate = True
            bounds = bound_axes(broadcast.src, broadcast.rcr, self.dx,
                                broadcast.est_z_max, broadcast.max_dur,
                                c=broadcast.c, to_rotate=True)
        else:
            self.to_rotate = False
            bounds = inner_bounds

        # setup x and y axes based on inner bounds
        xbounds = inner_bounds[0]
        n_start = xbounds[0] // self.dx
        Nx = int((xbounds[1] - xbounds[0]) / self.dx + 2)

        if Nx % 2: Nx += 1
        self.x_a = (np.arange(Nx) + n_start - 1) * self.dx

        ybounds = inner_bounds[1]
        n_start = ybounds[0] // self.dx
        Ny = int((ybounds[1] - ybounds[0]) / self.dx + 2)
        if Ny % 2: Ny += 1
        self.y_a = (np.arange(Ny) + n_start - 1) * self.dx


        self.x_img = broadcast.src[-1] * broadcast.rcr[0] \
                   / (broadcast.rcr[-1] + broadcast.src[-1])

        X, Y = np.meshgrid(self.x_a - self.x_img, self.y_a)
        self.rel_coords = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2)

        self.surface = Surface(bounds[0], bounds[1], kmax,
                               broadcast.toml_dict['surface'])

        self._generate_experiment(inner_bounds)



    def __call__(self, time_i, theta_i):
        """Return surface specified by time and theta indicies"""
        pass

    def _generate_experiment(self, inner_bounds):
        """pregenerate surfaces if needed"""

        t_file = self.broadcast.toml_file
        toml_time = os.path.getmtime(t_file)
        s_name = t_file.split('/')[-1].split('.')[0]
        save_dir = os.path.join('data/surfaces', s_name)
        spec_file = os.path.join(save_dir, s_name + '.txt')
        isdir = os.path.isdir(save_dir)

        if not isdir:
            os.mkdir(save_dir)
            to_run = True
        elif not os.path.isfile(spec_file):
            to_run = True
        else:
            with open(spec_file, "r") as f:
                timestamp = f.readline()
            1/0

        if not to_run:
            return

        # loop through a sequence of wave realizations"""
        realization = self.surface.realization()

        if self.surface.dt is None:
            # TODO: use this to generate new surface realizations
            1/0

        wave_time = np.arange(self.surface.num_snaps) \
                  * self.surface.dt

        for wt in wave_time:
            surf = self.surface.surface_synthesis(realization, time=wt)
            if self.to_rotate:
                surfaces = list(self.rotate_surface(surf))

    def rotate_surface(self, surface):
        """Rotate surface around specular reflection point"""
        ier = RectBivariateSpline(self.surface.x_a, self.surface.y_a, surface)

        for t in np.array(self.theta, ndmin=1):
            if abs(t) < 1e-6:
                grid_x_i = (self.surface.x_a >= self.x_a[0]) \
                         & (self.surface.x_a <= self.x_a[-1])
                grid_y_i = (self.surface.y_a >= self.y_a[0]) \
                         & (self.surface.y_a <= self.y_a[-1])
                rotated_grid = surface[np.ix_(grid_x_i, grid_y_i)]
                yield rotated_grid
            else:
                t_d = np.deg2rad(t)
                R = np.array([[np.cos(t_d), -np.sin(t_d)],
                              [np.sin(t_d), np.cos(t_d)]])
                x_shift = (self.x_a[:, None] - self.x_img) * R[0]
                y_shift = self.y_a[:, None] * R[1]
                # TODO: haven't gotten this quite right
                rot_coords = x_shift[:, None, :] + y_shift[None, :, :]

                rotated_surface = ier(rot_coords)

