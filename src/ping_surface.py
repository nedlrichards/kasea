import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src.specification import Broadcast
from src.helpers import bound_tau_ras

save_dir = 'data/canope'

class XMitt:
    """Run common setup and compute scatter time-series"""


    def __init__(self, toml_file, z_offset=1., num_sample_chunk=1e8):
        """Load xmission parameters and run basic setup"""
        experiment = Broadcast(toml_file, z_offset)
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


    def save(self, p_sca, time=None):
        """Save scattered pressure allong with toml meta data"""
        save_dict = copy.deepcopy(self.experiment.toml_dict)
        save_dict['p_sca'] = p_sca
        save_dict['time'] = time
        save_file = f"u20_{save_dict['U20']:.1f}.npz"
        np.savez(join(save_dir, save_file))


    def ping_surface(self, time=0.):
        """Compute a surface realization and compute scatter"""
        n_vec = self.surface_realization(time=time)
        src = self.experiment.src
        rcr = self.experiment.rcr
        c = self.experiment.c
        dx = self.experiment.surface.dx
        tau_img = self.experiment.tau_img

        f_a = self.f_a[:, None]
        pulse_FT = self.experiment.pulse_FT[:, None]

        # Kirchhoff approximation convolved with a source x_mission
        exper = "pulse_FT * projection " \
              + "* exp(-2j * pi * f_a * (tau_total - tau_img))" \
              + "* dx ** 2 / (8 * pi ** 2 * c * tau_total)"
        ka = np.zeros(pulse_FT.size, dtype=np.complex128)

        for x_i in self.x_inds:
            y_a = np.broadcast_to(self.y_a[None, :],
                              (x_i.size, self.y_a.size))
            x_a = np.broadcast_to(self.x_a[x_i, None],
                              (x_i.size, self.y_a.size))

            n = n_vec[:, x_i, :]
            a = np.array([x_a, y_a, n[2, :, :]])

            ras = a - src[:, None, None]
            ras_norm = np.linalg.norm(ras, axis=0)
            rra_norm = np.linalg.norm(rcr[:, None, None] - a, axis=0)

            tau_total = ne.evaluate('(ras_norm + rra_norm) / c')
            tau_mask = tau_total < self.experiment.t_max

            n = n[:, tau_mask]
            ras = ras[:, tau_mask]
            ras_norm = ras_norm[tau_mask]
            rra_norm = rra_norm[tau_mask]

            tau_total = tau_total[tau_mask]
            tau_total = tau_total[None, :]

            projection = np.sum(n * ras, axis=0) / ras_norm
            projection = projection[None, :]

            igrand = ne.evaluate(exper)
            ka += np.sum(igrand, axis=-1)

        return np.fft.irfft(ka)


    def surface_realization(self, time=0.):
        """Compute surface height and derivatives"""
        surf = self.experiment.surface
        rlz = self.realization
        out_size = (3, surf.x_a.size, surf.y_a.size)
        # make sure data is aligned in memory
        n_vec = np.empty(out_size)
        n_vec[0] = -surf.surface_synthesis(rlz, time=time, derivative='x')
        n_vec[1] = -surf.surface_synthesis(rlz, time=time, derivative='y')
        n_vec[2] = surf.surface_synthesis(rlz, time=time)
        return n_vec
