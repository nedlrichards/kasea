import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src import Broadcast, bound_tau_ras, Config
from src import ne_strs


class XMitt:
    """Run common setup and compute scatter time-series"""


    def __init__(self, toml_file, num_sample_chunk=5e6, save_dir=None):
        """Load xmission parameters and run basic setup"""
        if save_dir is not None:
            self.cf = Config(save_dir=save_dir)
        else:
            self.cf = Config()
        self.save_name = toml_file.split('/')[-1].split('.')[0]
        experiment = Broadcast(toml_file)
        self.num_sample_chunk = int(num_sample_chunk)

        self.fc = experiment.fc

        self.x_a = experiment.surface.x_a
        self.y_a = experiment.surface.y_a

        # receiver time axis
        self.t_a = experiment.t_a
        self.f_a = experiment.f_a

        # seperate time axis for surface sources
        self.surf_t_a = experiment.surf_t_a
        self.surf_f_a = experiment.surf_f_a
        self.dt = (self.t_a[-1] - self.t_a[0]) / (self.t_a.size - 1)

        self.tau_img = experiment.tau_img
        self.tau_max = experiment.tau_max

        self.experiment = experiment

        self.dx = (self.x_a[-1] - self.x_a[0]) / (self.x_a.size - 1)

        # make a rough estimate of number of processing chunks needed
        self.realization = None
        if self.y_a is None:
            self.src_type = '2D'
        else:
            self.src_type = '3D'
            self.dy = (self.y_a[-1] - self.y_a[0]) / (self.y_a.size - 1)


    def __call__(self):
        """loop through a sequence of wave realizations"""
        self.generate_realization()
        if self.experiment.surface.dt is None:
            1/0
        wave_time = np.arange(self.experiment.surface.num_snaps) \
                  * self.experiment.surface.dt

        p_sca = []
        for wt in wave_time:
            specs = self.setup(time=wt)
            ping = self.ping_surface(specs)
            p_sca.append(ping)
        p_sca = np.array(p_sca)

        self.save(p_sca, t_a_wave=wave_time)

        return wave_time, p_sca


    def generate_realization(self):
        """Generate new surface realization"""
        self.realization = self.experiment.surface.realization()


    def save(self, p_sca, t_a_wave=None):
        """Save scattered pressure allong with toml meta data"""
        save_dict = copy.deepcopy(self.experiment.toml_dict)
        save_dict['p_sca'] = p_sca
        save_dict['t_a_wave'] = t_a_wave
        np.savez(join(self.cf.save_dir, self.save_name))


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
        t_rcr_ref = self.tau_img + self.t_a[0]
        num_samp_shift = np.asarray((tau_ras - t_rcr_ref) / self.dt,
                                    dtype=np.int64)

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

        specs = {'t_rcr_ref':t_rcr_ref,
                 'num_samp_shift':num_samp_shift,
                 'i_scale':i_scale,
                 'dx_as':dx_as, 'dy_as':dy_as, 'dz_as':dz_as,
                 'dx_ra':dx_ra, 'dy_ra':dy_ra, 'dz_ra':dz_ra,
                 'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras}
        return specs


    def ping_surface(self, specs):
        """perform ka calculation over a single chunk"""
        # specification of ka integrand
        ka_str = ne_strs.dn_green_product(src_type=self.src_type)

        num_t_a = self.t_a.size
        ka = np.zeros(self.t_a.size + self.surf_t_a.size, dtype=np.float64)

        c = self.experiment.c
        f_a = self.surf_f_a[:, None]

        # sort by tau and then chunck computation

        nss = specs['num_samp_shift']
        nss_i = np.argsort(nss)

        pulse = self.experiment.pulse_FT[:, None]

        ka = np.zeros(self.t_a.size + self.surf_t_a.size, dtype=np.float64)
        i_start = 0
        i_next = self._next_ind(nss, nss_i, i_start)

        while i_next is not None:

            chunk = nss_i[i_start: i_next[-1]]

            proj = specs['proj'][chunk][None, :]
            tau_ras = specs['tau_ras'][chunk][None, :]
            n_vals = nss[chunk]
            D_tau = specs['num_samp_shift'][chunk][None, :] * self.dt
            tau_shift = specs['t_rcr_ref'] + D_tau
            m_as = specs['m_as'][chunk][None, :]
            m_ra = specs['m_ra'][chunk][None, :]

            ka_FT = ne.evaluate("pulse * " + ka_str)

            sum_inds = i_next - i_start
            last_i = 0

            for s_i in sum_inds:

                surf_FT = np.sum(ka_FT[:, last_i: s_i], axis=-1)
                surf_ts = np.fft.irfft(surf_FT)

                ka_i = n_vals[last_i]
                ka[ka_i: ka_i + surf_ts.size] += surf_ts

                last_i = s_i


            i_start = i_next[-1]
            i_next = self._next_ind(nss, nss_i, i_start)

        ka *= specs['i_scale']
        ka = ka[:self.t_a.size]
        return ka


    def _next_ind(self, nss, nss_i, i_start):
        """manage solution chunck size by index shift"""
        i_num = nss[nss_i[i_start]]

        j = 1
        i_max = min(i_start + j * self.num_sample_chunk, nss.size - 1)

        i_test = nss[nss_i[i_max]]
        while i_test == i_num:
            if i_max == nss.size - 1:
                break
            j += 1
            i_max = min(i_start + j * self.num_sample_chunk, nss.size - 1)
            i_test = nss[nss_i[i_max]]

        n_range = nss[nss_i[i_start :i_max]]

        end_off = 0
        inds = [0]
        for i in range(i_num + 1, i_test + 1):
            end_off = np.argmax(n_range[inds[-1]:] == i)
            inds.append(inds[-1] + end_off)

        # last index case
        if nss[nss_i[inds[-1]]] == nss[nss_i[-1]]:
            inds.append(nss.size - 1)

        if len(inds) == 1:
            return None

        inds = np.array(inds)[1:]

        return i_start + inds
