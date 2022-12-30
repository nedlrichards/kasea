import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from kasea import XMission, load_surface, Realization, integral_mask
from scipy.interpolate import RegularGridInterpolator

class Ping:
    """Run common setup and compute scatter time-series"""
    def __init__(self, toml_file, save_dir='results'):
        """Load xmission parameters and run basic setup"""
        self.save_dir = save_dir

        self.xmission = XMission(toml_file)
        self.surface = load_surface(self.xmission)
        include_hessian = False if self.xmission.solutions == ['2D'] else True
        self.realization = Realization(self.surface,
                                       include_hessian=include_hessian)

        self.save_name = self.surface.surface_type \
                       + f"_{self.surface.seed}"

    def one_time(self, time_step_num, surf_decimation=300):
        """Compute scattered pressure and save with meta data"""
        time = time_step_num * self.xmission.time_step

        #TODO: Make this call aware of solution type (maybe eigenrays?)
        self.realization.synthesize(time)
        eta = self.realization()

        # interpolators and higher derivatives required for stationary phase
        if eta.shape[0] == 6:
            x_a = self.surface.x_a
            y_a = self.surface.y_a

            eta_interp = RegularGridInterpolator((x_a, y_a), eta[0])
            e_dx_interp = RegularGridInterpolator((x_a, y_a), eta[1])
            e_dy_interp = RegularGridInterpolator((x_a, y_a), eta[2])
            e_dxdx_interp = RegularGridInterpolator((x_a, y_a), eta[3])
            e_dxdy_interp = RegularGridInterpolator((x_a, y_a), eta[4])
            e_dydy_interp = RegularGridInterpolator((x_a, y_a), eta[5])
            iers = [eta_interp, e_dx_interp, e_dy_interp, e_dxdx_interp,
                    e_dxdy_interp, e_dydy_interp]


        save_dict = {'t_a':self.xmission.t_a, 'time':time}

        rcr = []

        # 1D isotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['iso', 'all']):
            spec_KA = [spec for spec in self._iso_KA_byangle(*iers)]

        # 1D anisotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['aniso', 'all']):
            spec_KA = [spec for spec in self._aniso_KA_byangle(*iers)]

        # 2D surface results by angle
        if any(x in self.xmission.solutions for x in ['2D', 'all']):
            spec_KA = [spec for spec in self._2d_KA_byangle(eta)]
            p_sca_2D = np.array([self.ping_surface(spec) for spec in spec_KA])
            save_dict['p_sca_2D'] = p_sca_2D

        # save pressure time series and downsampled surface
        decimation = np.array(eta[0].shape) // surf_decimation

        save_dict['eta'] = eta[0, ::decimation[0], ::decimation[1]]
        save_dict['x_a'] = self.surface.x_a[::decimation[0]]
        save_dict['y_a'] = self.surface.y_a[::decimation[1]]
        save_dict['r_img'] = self.xmission.tau_img * self.xmission.c

        save_path = join(self.save_dir, self.save_name + f'_{time_step_num:03}.npz')
        np.savez(save_path, **save_dict)
        print('saved ' + save_path)
        return save_path


    def _iso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with isotropic surface"""
        z_src = self.xmission.z_src
        z_rcr = self.xmission.z_rcr

        for th in self.xmission.theta:
            1/0
            pass


    def _aniso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with 1D varing surface"""
        z_src = self.xmission.z_src
        z_rcr = self.xmission.z_rcr

        for th in self.xmission.theta:
            1/0
            pass


    def _2d_KA_byangle(self, eta):
        """Generate specifications of a scatter calculation"""
        z_src = self.xmission.z_src
        z_rcr = self.xmission.z_rcr

        for th in self.xmission.theta:
            igral_mask = integral_mask(self.realization, th, self.xmission)
            surface_height = eta[0][igral_mask]
            surface_dx = eta[1][igral_mask]
            (x_rcr, y_rcr) = self.xmission.dr * np.array([np.cos(th), np.sin(th)])
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

            #TODO: 1D calculation doesn't fit here
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
            m_as = ne.evaluate("sqrt(dx_as ** 2 + dy_as ** 2 + dz_as ** 2)")
            m_ra = ne.evaluate("sqrt(dx_ra ** 2 + dy_ra ** 2 + dz_ra ** 2)")

            # normal derivative projection
            proj = ne.evaluate("(-dx_as * surface_dx + dz_as - dy_as * surface_dy) / m_as")

            # time axis
            tau_img = self.xmission.tau_img
            tau_ras = (m_as + m_ra) / self.xmission.c

            # tau limit all arrays
            t_rcr_ref = tau_img + self.xmission.t_a[0]

            # TODO: this shouldn't be saved
            num_samp_shift = np.asarray((tau_ras - t_rcr_ref) / self.xmission.dt,
                                        dtype=np.int64)

            specs = {'t_rcr_ref':t_rcr_ref, 'num_samp_shift':num_samp_shift,
                    'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras,
                    'i_scale':i_scale, 'rcr':pos_rcr}
            yield specs
