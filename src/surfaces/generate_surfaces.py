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
        kmax = 2 * pi / broadcast.dx


        if 'theta' in broadcast.toml_dict['surface'] \
                and np.any(np.abs(broadcast.toml_dict['surface']['theta'])) > 1e-5:
            to_rotate = True
        else:
            to_rotate = False

        bounds = bound_axes(broadcast.src, broadcast.rcr, broadcast.dx,
                            broadcast.est_z_max, broadcast.max_dur,
                            c=broadcast.c, to_rotate=to_rotate)

        self.surface = Surface(bounds[0], bounds[1], kmax,
                               broadcast.toml_dict['surface'])

        self._generate_experiment()



    def __call__(self, time_i, theta_i):
        """Return surface specified by time and theta indicies"""
        pass


    def _generate_experiment(self):
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
            self.surface.rotate_surface(surf)
