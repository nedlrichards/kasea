import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from src import Broadcast, load_surface, Config, Realization, integral_mask

#string to compute distance from source to surface
m_as_str = "sqrt(dx_as ** 2  + dy_as ** 2+ dz_as ** 2)"

#string to compute distance from source to surface
m_ra_str = "sqrt(dx_ra ** 2 + dy_ra ** 2 + dz_ra ** 2)"

#string to compute n dot grad
proj_str = "(-dx_as * surface_dx + dz_as - dy_as * surface_dy) / m_as"

#string to compute product of two greens functions
dn_green_product_str = "proj * exp(-2j * pi * f_a * (tau_ras - tau_shift))"
dn_green_product_str +=  " * -1j * f_a / (4 * pi * c * m_as * m_ra)"

class XMitt:
    """Run common setup and compute scatter time-series"""
    def __init__(self, toml_file, num_sample_chunk=5e6, save_dir=None):
        """Load xmission parameters and run basic setup"""
        if save_dir is not None:
            self.cf = Config(save_dir=save_dir)
        else:
            self.cf = Config()

        self.num_sample_chunk = int(num_sample_chunk)

        self.broadcast = Broadcast(toml_file)
        self.surface = load_surface(self.broadcast)
        self.realization = Realization(self.surface)
        self.save_name = self.surface.surface_type \
                       + f"_{self.surface.seed}"


    def one_time(self, time_step_num, surf_decimation=300):
        """Compute scattered pressure and save with meta data"""
        time = time_step_num * self.broadcast.time_step
        self.realization.synthesize(time)
        eta = self.realization()

        save_dict = {'t_a':self.broadcast.t_a, 'time':time}

        rcr = []

        # 2D surface results by angle
        if any(x in self.broadcast.solutions for x in ['2D', 'all']):
            spec_2D_KA = [spec for spec in self._2d_KA_byangle(eta)]
            p_sca_2D = np.array([self.ping_surface(spec) for spec in spec_2D_KA])
            save_dict['p_sca_2D'] = p_sca_2D

        # save pressure time series and downsampled surface
        decimation = np.array(eta[0].shape) // surf_decimation


        save_dict['eta'] = eta[0, ::decimation[0], ::decimation[1]]
        save_dict['x_a'] = self.surface.x_a[::decimation[0]]
        save_dict['y_a'] = self.surface.y_a[::decimation[1]]
        save_dict['r_img'] = self.broadcast.tau_img * self.broadcast.c

        save_path = join(self.cf.save_dir, self.save_name + f'_{time_step_num:03}.npz')
        np.savez(save_path, **save_dict)
        print('saved ' + save_path)
        return save_path


    def _2d_KA_byangle(self, eta):
        """Generate specifications of a scatter calculation"""
        z_src = self.broadcast.z_src
        z_rcr = self.broadcast.z_rcr

        for th in self.broadcast.theta:
            igral_mask = integral_mask(self.realization, th, self.broadcast)
            surface_height = eta[0][igral_mask]
            surface_dx = eta[1][igral_mask]
            (x_rcr, y_rcr) = self.broadcast.dr * np.array([np.cos(th), np.sin(th)])
            pos_rcr = np.array([x_rcr, y_rcr, z_rcr])

            # x distances
            dx_as = self.surface.x_a
            dx_ra = x_rcr - self.surface.x_a

            if self.surface.y_a is not None:
                surface_dy = eta[2][igral_mask]

                shp = igral_mask.shape
                dy_as = np.broadcast_to(self.surface.y_a, shp)
                dy_ra = np.broadcast_to(y_rcr - self.surface.y_a, shp)

                # inflate x dimensions
                dx_as = np.broadcast_to(dx_as[:, None], shp)
                dx_ra = np.broadcast_to(dx_ra[:, None], shp)

                # apply surface mask
                dx_as = dx_as[igral_mask]
                dx_ra = dx_ra[igral_mask]
                dy_as = dy_as[igral_mask]
                dy_ra = dy_ra[igral_mask]

                i_scale = self.surface.dx ** 2
            else:
                # values required for ne string
                dy_as = 0.
                dy_ra = 0.
                surface_dy = 0.

                # apply surface mask
                dx_as = dx_as[igral_mask]
                dx_ra = dx_as[igral_mask]

                i_scale = self.surface.dx

            # isospeed delays to surface
            dz_as = surface_height - z_src
            dz_ra = z_rcr - surface_height

            # compute src and receiver distances
            m_as = ne.evaluate(m_as_str)
            m_ra = ne.evaluate(m_ra_str)

            # normal derivative projection
            proj = ne.evaluate(proj_str)

            # time axis
            tau_img = self.broadcast.tau_img
            tau_ras = (m_as + m_ra) / self.broadcast.c

            # tau limit all arrays
            t_rcr_ref = tau_img + self.broadcast.t_a[0]
            num_samp_shift = np.asarray((tau_ras - t_rcr_ref) / self.broadcast.dt,
                                        dtype=np.int64)

            specs = {'t_rcr_ref':t_rcr_ref, 'num_samp_shift':num_samp_shift,
                    'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras,
                    'i_scale':i_scale, 'rcr':pos_rcr}
            yield specs


    def ping_surface(self, specs):
        """perform ka calculation over a single chunk"""
        # specification of ka integrand
        t_a = self.broadcast.t_a
        f_a = self.broadcast.surf_f_a[:, None]
        c = self.broadcast.c

        num_t_a = t_a.size
        ka = np.zeros(num_t_a + self.broadcast.surf_t_a.size, dtype=np.float64)

        # sort by tau and then chunck computation
        nss = specs['num_samp_shift']
        nss_i = np.argsort(nss)

        pulse = self.broadcast.pulse_FT[:, None]

        i_start = 0
        i_next = self._next_ind(nss, nss_i, i_start)

        while i_next is not None:

            chunk = nss_i[i_start: i_next[-1]]

            proj = specs['proj'][chunk][None, :]
            tau_ras = specs['tau_ras'][chunk][None, :]
            n_vals = nss[chunk]
            D_tau = specs['num_samp_shift'][chunk][None, :] * self.broadcast.dt
            tau_shift = specs['t_rcr_ref'] + D_tau
            m_as = specs['m_as'][chunk][None, :]
            m_ra = specs['m_ra'][chunk][None, :]

            ka_FT = ne.evaluate("pulse * " + dn_green_product_str)

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
        ka = ka[:num_t_a]
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
