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


        save_path = join(self.save_dir, self.save_name + f'_{time_step_num:03}.npz')
        save_dict = {'t_a':self.xmission.t_a, 'time':time}

        if any(x in self.xmission.solutions for x in ['eigen', 'all']):
            specs = [spec for spec in self.eigen_KA_byangle(eta, *iers)]

        # 1D isotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['iso', 'all']):
            specs = [spec for spec in self.iso_KA_byangle(*iers)]

        # 1D anisotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['aniso', 'all']):
            specs = [spec for spec in self.aniso_KA_byangle(*iers)]

        # 2D surface results by angle
        if any(x in self.xmission.solutions for x in ['2D', 'all']):
            specs = [spec for spec in self.full_KA_byangle(eta)]
            #p_sca_2D = np.array([self.ping_surface(spec) for spec in spec_KA])
            #save_dict['p_sca_2D'] = p_sca_2D

        # save pressure time series and downsampled surface
        decimation = np.array(eta[0].shape) // surf_decimation

        save_dict['eta'] = eta[0, ::decimation[0], ::decimation[1]]
        save_dict['x_a'] = self.surface.x_a[::decimation[0]]
        save_dict['y_a'] = self.surface.y_a[::decimation[1]]
        save_dict['r_img'] = self.xmission.tau_img * self.xmission.c

        np.savez(save_path, **save_dict)
        print('saved ' + save_path)
        return save_path


    def _pos_rcr(self, theta):
        """Compute position of receiver from theta"""
        cs = np.array([np.cos(theta), np.sin(theta)])
        (x_rcr, y_rcr) = self.xmission.dr * cs
        pos_rcr = np.array([x_rcr, y_rcr, self.surface.z_rcr])
        return pos_rcr


    def _ray_geometry(self, eta, pos_rcr, surface_mask=None,
                      compute_derivatives=False):
        """Compute geometry quanities of KA kernel"""
        (x_rcr, y_rcr, z_rcr) = pos_rcr

        if surface_mask is not None:
            x = eta[0][surface_mask]
            y = eta[1][surface_mask]
            z = eta[2][surface_mask]
            dz_dx = eta[3][surface_mask]
            dz_dy = eta[4][surface_mask]
        else:
            x = eta[0]
            y = eta[1]
            z = eta[2]
            dz_dx = eta[3]
            dz_dy = eta[4]

        # source distances
        dx_s = x
        dy_s = y
        dz_s = z - self.surface.z_src

        # receiver distances
        dx_r = x - x_rcr
        dy_r = y - y_rcr
        dz_r = z - z_rcr

        # compute src and receiver distances
        m_s = ne.evaluate("sqrt(dx_s ** 2 + dy_s ** 2 + dz_s ** 2)")
        m_r = ne.evaluate("sqrt(dx_r ** 2 + dy_r ** 2 + dz_r ** 2)")

        # normal derivative projection
        proj = ne.evaluate("(-dx_s * dz_dx - dy_s * dz_dy + dz_s) / m_s")

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

        specs['hessian'] = hessian
        return specs


    def iso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with isotropic surface"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            eta = isotopic_igral(self.surface, th, eta_interp,
                                 e_dx_interp, e_dy_interp,
                                 e_dxdx_interp, e_dxdy_interp, e_dydy_interp)
            spec = self._ray_geometry(eta, pos_rcr, compute_derivatives=True)
            spec['i_scale'] = self.surface.dx
            yield spec


    def aniso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with 1D varing surface"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            eta = anisotopic_igral(self.surface, pos_rcr, eta_interp,
                                   e_dx_interp, e_dy_interp,
                                   e_dxdx_interp, e_dydy_interp)
            spec = self._ray_geometry(eta, pos_rcr, compute_derivatives=True)
            spec['i_scale'] = self.surface.dx
            yield spec


    def eigen_KA_byangle(self, eta, eta_interp, e_dx_interp, e_dy_interp,
                         e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """one bounce eigenray approximation of 2D surface scatter"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            points = stationary_points(self.surface, pos_rcr, eta, eta_interp,
                                       e_dx_interp, e_dy_interp, e_dxdx_interp,
                                       e_dxdy_interp, e_dydy_interp)
            spec = self._ray_geometry(points, pos_rcr, compute_derivatives=True)
            yield spec


    def full_KA_byangle(self, eta):
        """Generate specifications of a scatter calculation"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            igral_mask = integral_mask(self.realization, th, self.xmission)
            spec = self._ray_geometry(eta, pos_rcr, surface_mask=igral_mask)
            spec['i_scale'] = self.surface.dx ** 2
            yield spec
