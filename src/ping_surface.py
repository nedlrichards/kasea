import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src import Broadcast, bound_tau_ras, Config
from src import ne_strs


class XMitt:
    """Run common setup and compute scatter time-series"""


    def __init__(self, toml_file, num_sample_chunk=5e6):
        """Load xmission parameters and run basic setup"""
        self.cf = Config()
        experiment = Broadcast(toml_file)
        self.num_sample_chunk = num_sample_chunk

        self.x_a = experiment.surface.x_a
        self.y_a = experiment.surface.y_a

        # receiver time axis
        self.t_a = experiment.t_a
        self.f_a = experiment.f_a
        # seperate time axis for surface sources
        self.surf_t_a = experiment.t_a
        self.surf_f_a = experiment.f_a

        self.tau_img = experiment.tau_img
        self.tau_max = experiment.tau_max
        self.surf_tau_max = experiment.surf_tau_max

        self.experiment = experiment

        self.dx = (self.x_a[-1] - self.x_a[0]) / (self.x_a.size - 1)

        # make a rough estimate of number of processing chunks needed
        self.realization = None
        if self.y_a is None:
            self.src_type = '2D'
        else:
            self.src_type = '3D'
            self.dy = (self.y_a[-1] - self.y_a[0]) / (self.y_a.size - 1)


    def generate_realization(self):
        """Generate new surface realization"""
        self.realization = self.experiment.surface.realization()


    def save(self, file_name, p_sca, time=None):
        """Save scattered pressure allong with toml meta data"""
        save_dict = copy.deepcopy(self.experiment.toml_dict)
        save_dict['p_sca'] = p_sca
        save_dict['time'] = time
        np.savez(join(self.cf.save_dir, file_name))


    def setup(self, time=0.):
        """Compute a surface realization and compute scatter"""
        if self.src_type == '3D':
            (x_src, y_src, z_src) = self.experiment.src
            (x_rcr, y_rcr, z_rcr) = self.experiment.rcr
        else:
            (x_src, z_src) = self.experiment.src
            (x_rcr, z_rcr) = self.experiment.rcr

        # 1D distances
        dx_as = self.x_a - x_src
        dx_ra = x_rcr - self.x_a

        if self.src_type == '3D':
            dy_as = (self.y_a - y_src)[None, :]
            dy_ra = (y_rcr - self.y_a)[None, :]

            # inflate 1D dimensions
            dx_as = dx_as[:, None]
            dx_ra = dx_ra[:, None]
            i_scale = self.dx * self.dy
        else:
            i_scale = self.dx

        # isospeed delays to surface
        surface = self.experiment.surface
        surface_height = surface.surface_synthesis(self.realization, time=time)
        surface_dx = surface.surface_synthesis(self.realization,
                                               derivative='x', time=time)
        if surface.y_a is not None:
            surface_dy = surface.surface_synthesis(self.realization,
                                                   derivative='y', time=time)

        dz_as = surface_height - z_src
        dz_ra = z_rcr - surface_height

        # compute src and receiver distances
        m_as = ne.evaluate(ne_strs.m_as(self.src_type))
        m_ra = ne.evaluate(ne_strs.m_ra(self.src_type))

        # normal derivative projection
        proj = ne.evaluate(ne_strs.proj(src_type=self.src_type))

        # time axis
        tau_img = self.experiment.tau_img
        tau_ras = (m_as + m_ra) / self.experiment.c
        # bound integration by delay time
        tau_lim = self.experiment.tau_max
        tau_i = ne.evaluate("tau_ras < tau_lim")

        # tau limit all arrays
        tau_ras = tau_ras[tau_i]

        dx_as = np.broadcast_to(dx_as, m_as.shape)[tau_i]
        dx_ra = np.broadcast_to(dx_ra, m_as.shape)[tau_i]

        if self.src_type == "3D":
            dy_as = np.broadcast_to(dy_as, m_as.shape)[tau_i]
            dy_ra = np.broadcast_to(dy_ra, m_as.shape)[tau_i]
        else:
            dy_as = None
            dy_ra = None

        dz_as = dz_as[tau_i]
        dz_ra = dz_ra[tau_i]

        m_as = m_as[tau_i]
        m_ra = m_ra[tau_i]

        proj = proj[tau_i]

        specs = {'inds':np.arange(tau_i.sum()), 'i_scale':i_scale,
                 'dx_as':dx_as, 'dy_as':dy_as, 'dz_as':dz_as,
                 'dx_ra':dx_ra, 'dy_ra':dy_ra, 'dz_ra':dz_ra,
                 'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras}
        return specs


    def ping_surface(self, specs):
        """perform ka calculation over a single chunk"""
        ka = np.zeros(self.f_a.size, dtype=np.complex128)

        f_a = self.f_a[None, :]
        i_scale = specs['i_scale']
        inds = specs['inds']
        c = self.experiment.c
        num_chunks = np.ceil(inds.size / self.num_sample_chunk)

        chunk_inds = np.array_split(inds, num_chunks)
        ka_str = ne_strs.dn_green_product(src_type=self.src_type)

        for chunk in chunk_inds:
            dx_as = specs['dx_as'][chunk][:, None]
            dx_ra = specs['dx_ra'][chunk][:, None]
            if self.src_type == "3D":
                dy_as = specs['dy_as'][chunk][:, None]
                dy_ra = specs['dy_ra'][chunk][:, None]
            dz_as = specs['dz_as'][chunk][:, None]
            dz_ra = specs['dz_ra'][chunk][:, None]
            m_as = specs['m_as'][chunk][:, None]
            m_ra = specs['m_ra'][chunk][:, None]
            proj = specs['proj'][chunk][:, None]
            tau_ras = specs['tau_ras'][chunk][:, None]

            ka += np.sum(ne.evaluate(ka_str), axis=0)
        ka *= i_scale

        # scale and shift td
        f_shift = np.exp(2j * pi * self.f_a * \
                (self.experiment.tau_img - self.t_a[0]))

        return np.fft.irfft(ka * self.experiment.pulse_FT * f_shift)
