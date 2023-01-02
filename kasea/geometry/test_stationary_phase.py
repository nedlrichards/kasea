import numpy as np
from math import pi
import numexpr as ne
import matplotlib.pyplot as plt
from kasea import Ping

import pytest
from scipy.misc import central_diff_weights
from kasea import XMission
from kasea.geometry.stationary_phase import anisotopic_igral
from kasea.surfaces import load_surface

plt.ion()

flat_toml = 'experiments/flat_surface.toml'

class TestStationaryPhase:
    def __init__(self):
        self.xmission = XMission(flat_toml)
        self.surface = load_surface(self.xmission)
        self.theta_test = np.deg2rad(45.)
        self.ier = lambda x: np.zeros(x.shape[0])

    def test_anisotopic_igral(self):
        position = anisotopic_igral(self.surface,
                                    self.theta_test,
                                    *(5 * [self.ier]))[1]

        # finite difference estimate of position
        x_r = self.surface.dr * np.cos(self.theta_test)
        y_r = self.surface.dr * np.sin(self.theta_test)
        x = self.surface.x_a[:, None]
        y = self.surface.y_a[None, :]
        z_src = self.surface.z_src
        z_rcr = self.surface.z_rcr
        r_dy = y / np.sqrt(x ** 2 + y ** 2 + z_src ** 2) \
             + (y - y_r) / np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2 + z_rcr ** 2)
        fd_i = np.argmin(np.abs(r_dy), axis=-1)
        y_est = self.surface.y_a[fd_i]

        assert(np.max(np.abs(position - y_est)) / self.surface.dx < 0.5)

ping = Ping(flat_toml)
ping.one_time(0.)
