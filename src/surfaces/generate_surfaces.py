import tomli
import numpy as np
import numexpr as ne
from math import pi
import sys
import importlib.util
import os

from src.surfaces import Surface
from src.helpers import bound_axes

class SurfGen:
    """prepare surfaces specified by a test scenario"""


    def __init__(self, broadcast):
        """scatter calculation specification load and basic setup"""
        self.broadcast = broadcast
        self.dx = broadcast.dx
        kmax = 2 * pi / self.dx

        inner_bounds =  bound_axes(broadcast.src, broadcast.rcr, broadcast.dx,
                                   broadcast.est_z_max, broadcast.max_dur,
                                   c=broadcast.c, to_rotate=False)

        if 'theta' in broadcast.toml_dict['surface'] \
                and np.any(np.abs(broadcast.toml_dict['surface']['theta'])) > 1e-5:
            self.to_rotate = True
            bounds = bound_axes(broadcast.src, broadcast.rcr, self.dx,
                                broadcast.est_z_max, broadcast.max_dur,
                                c=broadcast.c, to_rotate=True)
        else:
            self.to_rotate = False
            bounds = inner_bounds

        # setup x and y axes based on inner bounds
        xbounds = inner_bounds[0]
        Nx = int((xbounds[1] - xbounds[0]) / self.dx + 2)

        if Nx % 2: Nx += 1
        x_i = np.arange(Nx)
        self.x_a = x_i * self.dx + xbounds[0] - self.dx

        ybounds = inner_bounds[1]
        Ny = int((ybounds[1] - ybounds[0]) / self.dx + 2)
        if Ny % 2: Ny += 1
        y_i = np.arange(Ny)
        self.y_a = y_i * self.dx + ybounds[0] - self.dx


        self.x_img = broadcast.src[-1] * broadcast.rcr[0] \
                   / (broadcast.rcr[-1] + broadcast.src[-1])

        self.rel_coords = np.array(np.meshgrid(self.x_a - self.x_img, self.y_a))

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
                self.rotate_surface(surf)

    def rotate_surface(self, surface):
        """Rotate surface around specular reflection point"""
        ier = RectBivariateSpline(self.surface.x_a, self.surface.y_a, surface)

        for t in np.array(self.theta, ndmin=1):
            if abs(t) < 1e-6:
                1/0
                return rotated_grid
            t = np.deg2rad(theta)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

            rot_coords = R @ self.rel_coords
            rotated_surface = ier(rot_coords)

