import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src import Broadcast, bound_tau_ras, Config


class XMitt:
    """Run common setup and compute scatter time-series"""


    def __init__(self, toml_file, num_sample_chunk=1e8):
        """Load xmission parameters and run basic setup"""
        self.cf = Config()
        experiment = Broadcast(toml_file)
        self.num_sample_chunk = num_sample_chunk

        self.x_a = experiment.surface.x_a
        self.y_a = experiment.surface.y_a
        self.t_a = experiment.t_a
        self.f_a = experiment.f_a
        self.experiment = experiment

        # make a rough estimate of number of processing chunks needed
        num_samples = self.x_a.size * self.y_a.size * self.f_a.size
        num_chunks = int(np.ceil(num_samples / num_sample_chunk))

        self.x_inds = np.array_split(np.arange(self.x_a.size), num_chunks)
        self.realization = None


    def generate_realization(self):
        """Generate new surface realization"""
        self.realization = self.experiment.surface.realization()


    def save(self, file_name, p_sca, time=None):
        """Save scattered pressure allong with toml meta data"""
        save_dict = copy.deepcopy(self.experiment.toml_dict)
        save_dict['p_sca'] = p_sca
        save_dict['time'] = time
        np.savez(join(self.cf.save_dir, file_name))


    def ping_surface(self, time=0.):
        """Compute a surface realization and compute scatter"""

        src = self.experiment.src
        rcr = self.experiment.rcr
        x_a = self.x_a
        x_src = src[0]
        c = self.experiment.c
        dx = self.experiment.surface.dx

        # isospeed delays to surface
        surface_height = self.surface_realization(time=time)
        dx_as = self.x_a - src[0]
        dx_ra = rcr[0] - self.x_a

        dz_as = surface_height - src[-1]
        dz_ra = rcr[-1] - surface_height

        surface_dx = -self.surface_realization(derivative='x', time=time)
        if self.surface.y_a is not None:
            dy_as = self.y_a - src[1]
            dy_ra = rcr[1] - self.y_a

            surface_dy = -self.surface_realization(derivative='y', time=time)
            proj_str = "dx_as * surface_dx + dy_as * surface_dy"
            d_as_str = "dx_as ** 2 + dy_as ** 2"
            # TODO: put ra differences in ne str?
            d_ra_str = "dx_ra ** 2 + dy_ra ** 2"

        else:
            proj_str = "dx_as * surface_dx"
            d_as_str = "dx_as ** 2"
            d_ra_str = "dx_ra ** 2"

        proj = ne.evaluate(proj_str + " + dz_as")
        d_as = ne.evaluate("sqrt(" + d_as_str + "dz_as ** 2)")
        d_ra = ne.evaluate("sqrt(" + d_ra_str + "dz_ra ** 2)")

        tau_ras = ne.evaluate("(d_as + d_ra) / c")

        # time and frequency axes
        f_a = self.f_a[:, None]
        t_a = self.t_a[:, None]
        tau_img = self.experiment.tau_img
        pulse_FT = self.experiment.pulse_FT[:, None]

        # Kirchhoff approximation convolved with a source x_mission

        front = "pulse_FT * projection / d_as"
        phase = "exp(-2j * pi * f_a * (tau_ras - tau_img))"
        spreading = "d_as * d_ra"
        scale = "dx ** 2 / (8 * pi ** 2)"
        # TODO: what axis is the summation?
        1/0

        return np.fft.irfft(ka)


