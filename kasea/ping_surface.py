import numpy as np
import numexpr as ne
from math import pi
from os.path import join
import copy

from kasea import XMission, load_surface, Realization, integral_mask
from kasea import anisotopic_igral, isotopic_igral, stationary_points

from scipy.interpolate import RegularGridInterpolator

class Ping:
    """Run common setup and compute scatter time-series"""
    def __init__(self, toml_file, save_dir='results'):
        """Load xmission parameters and run basic setup"""
        self.save_dir = save_dir

        self.xmission = XMission(toml_file)
        self.surface = load_surface(self.xmission)
        self.realization = Realization(self.surface)

        self.save_name = self.surface.surface_type \
                       + f"_{self.surface.seed}"

    def one_time(self, time_step_num, surf_decimation=300):
        """Compute scattered pressure and save with meta data"""
        time = time_step_num * self.xmission.time_step

        #TODO: Make this call aware of solution type (maybe eigenrays?)
        self.realization.synthesize(time)
        eta = self.realization()

        # interpolators and higher derivatives required for stationary phase
        x_a = self.surface.x_a
        y_a = self.surface.y_a

        eta_interp = RegularGridInterpolator((x_a, y_a), eta[0],
                                                bounds_error=False)
        e_dx_interp = RegularGridInterpolator((x_a, y_a), eta[1],
                                                bounds_error=False)
        e_dy_interp = RegularGridInterpolator((x_a, y_a), eta[2],
                                                bounds_error=False)
        e_dxdx_interp = RegularGridInterpolator((x_a, y_a), eta[3],
                                                bounds_error=False)
        e_dxdy_interp = RegularGridInterpolator((x_a, y_a), eta[4],
                                                bounds_error=False)
        e_dydy_interp = RegularGridInterpolator((x_a, y_a), eta[5],
                                                    bounds_error=False)
        iers = [eta_interp, e_dx_interp, e_dy_interp, e_dxdx_interp,
                e_dxdy_interp, e_dydy_interp]


        save_dict = {'t_a':self.xmission.t_a, 'time':time}

        rcr = []

        if any(x in self.xmission.solutions for x in ['eigen', 'all']):
            specs = [spec for spec in self._eigen_KA_byangle(eta, *iers)]

        # 1D isotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['iso', 'all']):
            for spec in self._iso_KA_byangle(*iers):
                save_dict['iso_x_a'] = spec['x_a']
                save_dict['iso_y_a'] = spec['y_a']

        # 1D anisotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['aniso', 'all']):
            for spec in self._aniso_KA_byangle(*iers):
                save_dict['aniso_x_a'] = spec['x_a']
                save_dict['aniso_y_a'] = spec['y_a']

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


    def _pos_rcr(self, theta):
        """Compute position of receiver from theta"""
        cs = np.array([np.cos(theta), np.sin(theta)])
        (x_rcr, y_rcr) = self.xmission.dr * cs
        pos_rcr = np.array([x_rcr, y_rcr, z_rcr])
        return pos_rcr


    def _ray_geometry(self, eta, pos_rcr, surface_mask=False,
                      compute_derivatives=False):
        """Compute geometry quanities of KA kernel"""
        (x_rcr, y_rcr, z_rcr) = pos_rcr

        if surface_mask:
            z = eta[2][surface_mask]
            dz_dx = eta[3][surface_mask]
            dz_dy = eta[4][surface_mask]
        else:
            z = eta[2]
            dz_dx = eta[3]
            dz_dy = eta[4]

        # source distances
        dx_s = self.surface.x_a
        dy_s = self.surface.y_a
        dz_s = z - self.surface.z_src

        # receiver distances
        dx_r = self.surface.x_a - x_rcr
        dy_r = self.surface.y_a - y_rcr
        dz_r = z - z_rcr

        # compute src and receiver distances
        m_s = ne.evaluate("sqrt(dx_as ** 2 + dy_as ** 2 + dz_as ** 2)")
        m_r = ne.evaluate("sqrt(dx_ra ** 2 + dy_ra ** 2 + dz_ra ** 2)")

        # normal derivative projection
        proj = ne.evaluate("(-dx_s * dz_dx - dy_s * dz_dy + dz_as) / m_s")

        # time delay
        tau_ras = (m_s + m_r) / self.xmission.c

        specs = {'proj':proj, 'm_as':m_s, 'm_ra':m_r,
                 'tau_ras':tau_ras, 'rcr':pos_rcr}

        if not compute_derivatives:
            return specs

        # hessian calculation
        dz_dxdx = eta[5]
        dz_dxdy = eta[6]
        dz_dydy = eta[7]

        dr_dx_s = (dx_s + dz_s * dz_dx) / m_s
        dr_dy_s = (dy_s + dz_s * dz_dy) / m_s
        dr_dx_r = (dx_r + dz_s * dz_dx) / m_r
        dr_dy_r = (dy_r + dz_s * dz_dy) / m_r

        dr_dxdx_s = ((1 + dz_dxdx * dz_s + dz_dx ** 2) - dr_dx_s ** 2) / m_s
        dr_dxdx_r = ((1 + dz_dxdx * dz_r + dz_dx ** 2) - dr_dx_r ** 2) / m_r
        dr_dxdy_s = (dz_dxdy * dz_s + dr_dx_s * dr_dy_s) / m_s
        dr_dxdy_r = (dz_dxdy * dz_r + dr_dx_r * dr_dy_r) / m_r
        dr_dydy_s = ((1 + dz_dydy * dz_s + dz_dy ** 2) - dr_dy_s ** 2) / m_s
        dr_dydy_r = ((1 + dz_dydy * dz_r + dz_dy ** 2) - dr_dy_r ** 2) / m_r

        hessian_1 = np.concatenate(((dr_dxdx_s + dr_dxdx_r)[:, None, None],
                                    (dr_dxdy_s + dr_dxdy_r)[:, None, None]),
                                    axis=1)
        hessian_2 = np.concatenate(((dr_dxdy_s + dr_dxdy_r)[:, None, None],
                                    (dr_dydy_s + dr_dydy_r)[:, None, None]),
                                    axis=1)
        hessian = np.concatenate([hessian_1, hessian_2], axis=2)

        specs{'hessian'} = hessian
        return specs






    def _iso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with isotropic surface"""
        for th in self.xmission.theta:
            surface = isotopic_igral(self.surface, th, eta_interp,
                                     e_dx_interp, e_dy_interp,
                                     e_dxdx_interp, e_dydy_interp)

            1/0
            #TODO: This save is nonsense
            specs = {'x_a':surface[0], 'y_a':surface[1]}
            yield specs


    def _aniso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with 1D varing surface"""
        for th in self.xmission.theta:
            surface = anisotopic_igral(self.surface, th, eta_interp,
                                       e_dx_interp, e_dy_interp,
                                       e_dxdx_interp, e_dydy_interp)
            1/0
            #TODO: This save is nonsense
            specs = {'x_a':surface[0], 'y_a':surface[1]}
            yield specs


    def _eigen_KA_byangle(self, eta, eta_interp, e_dx_interp, e_dy_interp,
                         e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """one bounce eigenray approximation of 2D surface scatter"""
        z_src = self.xmission.z_src
        z_rcr = self.xmission.z_rcr

        for th in self.xmission.theta:
            (x_rcr, y_rcr) = self.xmission.dr * np.array([np.cos(th), np.sin(th)])
            pos_rcr = np.array([x_rcr, y_rcr, z_rcr])
            points = stationary_points(self.surface, th, eta, eta_interp,
                                       e_dx_interp, e_dy_interp, e_dxdx_interp,
                                       e_dxdy_interp, e_dydy_interp)
            dx_s = points[:, 0]
            dy_s = points[:, 1]
            dz_s = (points[:, 2] - z_src)
            dx_r = (points[:, 0] - x_rcr)
            dy_r = (points[:, 1] - y_rcr)
            dz_r = (points[:, 2] - z_rcr)

            d_s = np.sqrt(dx_s ** 2 + dy_s ** 2 + dz_s ** 2)
            d_r = np.sqrt(dx_r ** 2 + dy_r ** 2 + dz_r ** 2)

            dr_dx_s = (dx_s + dz_s * points[:, 3]) / d_s
            dr_dy_s = (dy_s + dz_s * points[:, 4]) / d_s
            dr_dx_r = (dx_r + dz_s * points[:, 3]) / d_r
            dr_dy_r = (dy_r + dz_s * points[:, 4]) / d_r

            dr_dxdx_s = ((1 + points[:, 5] * dz_s + points[:, 3] ** 2) - dr_dx_s ** 2) / d_s
            dr_dxdx_r = ((1 + points[:, 5] * dz_r + points[:, 3] ** 2) - dr_dx_r ** 2) / d_r
            dr_dxdy_s = (points[:, 6] * dz_s + dr_dx_s * dr_dy_s) / d_s
            dr_dxdy_r = (points[:, 6] * dz_r + dr_dx_r * dr_dy_r) / d_r
            dr_dydy_s = ((1 + points[:, 7] * dz_s + points[:, 4] ** 2) - dr_dy_s ** 2) / d_s
            dr_dydy_r = ((1 + points[:, 7] * dz_r + points[:, 4] ** 2) - dr_dy_r ** 2) / d_r

            # computed quanitites
            tau = (d_s + d_r) / self.surface.c

            hessian_1 = np.concatenate(((dr_dxdx_s + dr_dxdx_r)[:, None, None],
                                        (dr_dxdy_s + dr_dxdy_r)[:, None, None]),
                                        axis=1)
            hessian_2 = np.concatenate(((dr_dxdy_s + dr_dxdy_r)[:, None, None],
                                        (dr_dydy_s + dr_dydy_r)[:, None, None]),
                                        axis=1)
            hessian = np.concatenate([hessian_1, hessian_2], axis=-1)

            surface_dx = points[:, 3]
            surface_dy = points[:, 4]

            proj = ne.evaluate("(-dx_s * surface_dx + dz_s - dy_s * surface_dy) / d_s")

            specs = {'x':dx_s, 'y':dy_s, 'z':points[:, 2],
                     'pos_rcr':pos_rcr, 'proj':proj, 'm_as':d_s, 'm_ra':d_r,
                     'tau_ras':tau, 'hessian':hessian}

            return specs


    def _2d_KA_byangle(self, eta):
        """Generate specifications of a scatter calculation"""
        z_src = self.xmission.z_src
        z_rcr = self.xmission.z_rcr

        i_scale = self.surface.dx ** 2

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
            # TODO: these shouldn't be saved
            t_rcr_ref = tau_img + self.xmission.t_a[0]
            num_samp_shift = np.asarray((tau_ras - t_rcr_ref) / self.xmission.dt,
                                        dtype=np.int64)

            specs = {'t_rcr_ref':t_rcr_ref, 'num_samp_shift':num_samp_shift,
                    'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras,
                    'i_scale':i_scale, 'rcr':pos_rcr}
            yield specs
