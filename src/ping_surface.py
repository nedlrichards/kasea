import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src import Broadcast, bound_tau_ras, Config


class XMitt:
    """Run common setup and compute scatter time-series"""


    def __init__(self, toml_file, num_sample_chunk=1e6):
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
        surface = self.experiment.surface

        # isospeed delays to surface
        surface_height = surface.surface_synthesis(self.realization, time=time)
        dx_as = self.x_a - src[0]
        dx_ra = rcr[0] - self.x_a

        dz_as = surface_height - src[-1]
        dz_ra = rcr[-1] - surface_height

        surface_dx = -surface.surface_synthesis(self.realization,
                                                derivative='x', time=time)

        if surface.y_a is not None:
            dy_as = self.y_a - src[1]
            dy_ra = rcr[1] - self.y_a

            dx_as = dx_as[:, None]
            dx_ra = dx_ra[:, None]
            dy_ra = dy_ra[None, :]

            surface_dy = -surface.surface_synthesis(self.realization,
                                                    derivative='y', time=time)
            proj_str = "dx_as * surface_dx + dy_as * surface_dy"
            d_as_str = "dx_as ** 2 + dy_as ** 2"
            # TODO: put ra differences in ne str?
            d_ra_str = "dx_ra ** 2 + dy_ra ** 2"

        else:
            proj_str = "dx_as * surface_dx"
            d_as_str = "dx_as ** 2"
            d_ra_str = "dx_ra ** 2"

        proj = ne.evaluate(proj_str + " + dz_as")
        d_as = ne.evaluate("sqrt(" + d_as_str + "+ dz_as ** 2)")
        d_ra = ne.evaluate("sqrt(" + d_ra_str + "+ dz_ra ** 2)")

        tau_ras = ne.evaluate("(d_as + d_ra) / c")

        # time and frequency axes
        tau_img = self.experiment.tau_img

        # bound integration by delay time
        tau_lim = self.experiment.tau_max
        tau_i = ne.evaluate("tau_ras < tau_lim")

        proj = proj[tau_i]
        tau_ras = tau_ras[tau_i]
        d_as = d_as[tau_i]
        d_ra = d_ra[tau_i]

        num_chunks = np.ceil(tau_i.sum() / self.num_sample_chunk)
        1/0


        # Kirchhoff approximation convolved with a source x_mission

        front = "pulse_FT * proj / d_as"
        phase = "exp(-2j * pi * f_a * (tau_ras - tau_img))"
        spreading = "d_as * d_ra"
        scale = "dx ** 2 / (8 * pi ** 2)"

        ka = ne.evaluate(front + '*' + phase + '*' + spreading + '*' + scale)
        # TODO: what axis is the summation?
        if surface.y_a is not None:
            1/0
        ka = ka.sum(axis=-1)

        return np.fft.irfft(ka)

    def _chunk_process(self, proj, tau_ras, d_as, d_ra, num_chunks):
        """perform ka calculation over a single chunk"""
        # setup ne calculations
        tau_img = self.experiment.tau_img
        dx = self.experiment.surface.dx
        pulse_FT = self.experiment.pulse_FT[:, None]
        f_a = self.f_a[:, None]

        # chunk array
        p_c = np.array_spit(proj, num_chunks)
        tau_c = np.array_spit(tau_ras, num_chunks)
        d_as_c = np.array_spit(d_as, num_chunks)
        d_ra = np.array_spit(d_ra, num_chunks)


        ier = zip(np.array_spit(proj, num_chunks),
                  np.array_spit(tau_ras, num_chunks),
                  np.array_spit(d_as, num_chunks),
                  np.array_spit(d_ra, num_chunks))

        front = "pulse_FT * p_c / d_as_c"
        phase = "exp(-2j * pi * f_a * (tau_c - tau_img))"
        spreading = "d_as_c * d_ra_c"
        scale = "dx ** 2 / (8 * pi ** 2)"
        ne_str = front + '*' + phase + '*' + spreading + '*' + scale
        ka = ne.evaluate(ne_str).sum(axis=-1)

        for (p_c, tau_c , d_as_c, d_ra) in ier:
            pass
            #ka += ne.evaluate(ne_str).sum(axis=-1)
        ka = ne.evaluate(ne_str).sum(axis=-1)

