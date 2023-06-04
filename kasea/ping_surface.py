import numpy as np
import numexpr as ne
from math import pi
from os.path import join
from os import makedirs
import copy
from scipy.linalg import eigvals
from time import gmtime, strftime

from kasea import XMission, load_surface, Realization, spec_igral_FT, spec_igral
from kasea.geometry import anisotopic_igral, isotopic_igral, \
                           stationary_points, integral_mask

from scipy.interpolate import RegularGridInterpolator

class Ping:
    """Run common setup and compute scatter time-series"""
    def __init__(self, toml_file, save_dir='results', ier='FT'):
        """Load xmission parameters and run basic setup"""
        run_name = toml_file.split('/')[-1].split('.')[0]
        self.save_dir = join(save_dir, run_name)
        makedirs(self.save_dir, exist_ok=True)

        self.xmission = XMission(toml_file)
        self.surface = load_surface(self.xmission)
        self.realization = Realization(self.surface)
        self.igral = spec_igral_FT if ier == 'FT' else spec_igral

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
            p = [self.igral(self.xmission, spec) for spec in specs]
            for i, spec in enumerate(specs):
                save_dict[f'sta_points_{i:03}'] = spec['sta_points']
            save_dict['p_sca_eig'] = np.array(p)


        # 1D isotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['iso', 'all']):
            specs = [spec for spec in self.iso_KA_byangle(*iers)]
            p = [self.igral(self.xmission, spec) for spec in specs]
            save_dict['p_sca_iso'] = np.array(p)

        # 1D anisotropic surface results by angle
        if any(x in self.xmission.solutions for x in ['aniso', 'all']):
            specs = [spec for spec in self.aniso_KA_byangle(*iers)]
            p = [self.igral(self.xmission, spec) for spec in specs]
            save_dict['p_sca_ani'] = np.array(p)

        # 2D surface results by angle
        if any(x in self.xmission.solutions for x in ['2D', 'all']):
            specs = [spec for spec in self.full_KA_byangle(eta)]
            p = [self.igral(self.xmission, spec) for spec in specs]
            save_dict['p_sca_2D'] = np.array(p)

        # save pressure time series and downsampled surface
        decimation = np.array(eta[0].shape) // surf_decimation

        save_dict['eta'] = eta[0, ::decimation[0], ::decimation[1]]
        save_dict['x_a'] = self.surface.x_a[::decimation[0]]
        save_dict['y_a'] = self.surface.y_a[::decimation[1]]
        save_dict['r_img'] = self.xmission.tau_img * self.xmission.c
        save_dict['theta'] = self.xmission.theta

        for i, th in enumerate(self.xmission.theta):
            save_dict[f'pos_rcr_{i:03}'] = self._pos_rcr(th)

        # image solution
        p_img = np.zeros(self.xmission.t_a.size)
        sample_shift = -self.xmission.t_a[0] / self.xmission.dt
        img_FT = -self.xmission.pulse_FT \
               * np.exp(-2j * pi * self.xmission.f_a_pulse * (sample_shift % 1))
        img_ts = np.fft.irfft(img_FT)
        s = int(sample_shift // 1)
        if s < 0: 1/0  # Shouldn't have negative ined shifts
        p_img[s: s + self.xmission.t_a_pulse.size] = img_ts
        save_dict['p_img'] = p_img / (4 * pi * save_dict['r_img'])

        np.savez(save_path, **save_dict)
        print('saved ' + save_path + ' at ' + strftime("%H:%M:%S", gmtime()))
        return save_path


    def _pos_rcr(self, theta):
        """Compute position of receiver from theta"""
        cs = np.array([np.cos(theta), np.sin(theta)])
        (x_rcr, y_rcr) = self.xmission.dr * cs
        pos_rcr = np.array([x_rcr, y_rcr, self.surface.z_rcr])
        return pos_rcr


    def ray_geometry(self, eta, pos_rcr, surface_mask=None,
                      compute_derivatives=False):
        """Compute geometry quanities of KA kernel"""
        (x_rcr, y_rcr, z_rcr) = pos_rcr

        x = eta[0]
        y = eta[1]
        z = eta[2]
        dz_dx = eta[3]
        dz_dy = eta[4]

        if surface_mask is not None:
            x = x[surface_mask]
            y = y[surface_mask]
            z = z[surface_mask]
            dz_dx = dz_dx[surface_mask]
            dz_dy = dz_dy[surface_mask]

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
        c = self.xmission.c
        tau_ras = (m_s + m_r) / c

        specs = {'proj':proj, 'm_as':m_s, 'm_ra':m_r,
                 'tau_ras':tau_ras, 'rcr':pos_rcr}

        if not compute_derivatives:
            return specs

        # hessian calculation
        dz_dxdx = eta[5]
        dz_dxdy = eta[6]
        dz_dydy = eta[7]

        dr_dx_s = ne.evaluate("(dx_s + dz_s * dz_dx) / m_s")
        dr_dy_s = ne.evaluate("(dy_s + dz_s * dz_dy) / m_s")
        dr_dx_r = ne.evaluate("(dx_r + dz_s * dz_dx) / m_r")
        dr_dy_r = ne.evaluate("(dy_r + dz_s * dz_dy) / m_r")

        dr_dxdx_s = ne.evaluate("((1 + dz_dxdx * dz_s + dz_dx ** 2) - dr_dx_s ** 2) / m_s")
        dr_dxdx_r = ne.evaluate("((1 + dz_dxdx * dz_r + dz_dx ** 2) - dr_dx_r ** 2) / m_r")
        dr_dxdy_s = ne.evaluate("(dz_dxdy * dz_s + dr_dx_s * dr_dy_s) / m_s")
        dr_dxdy_r = ne.evaluate("(dz_dxdy * dz_r + dr_dx_r * dr_dy_r) / m_r")
        dr_dydy_s = ne.evaluate("((1 + dz_dydy * dz_s + dz_dy ** 2) - dr_dy_s ** 2) / m_s")
        dr_dydy_r = ne.evaluate("((1 + dz_dydy * dz_r + dz_dy ** 2) - dr_dy_r ** 2) / m_r")

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

            # Rotate surface derivatives
            rotation = np.array(([np.cos(th), -np.sin(th)],
                                 [np.sin(th), np.cos(th)]))
            e_1 = np.concatenate((eta[3][:, None, None],
                                  eta[4][:, None, None]),
                                  axis=1)
            eta_ders = rotation @ e_1
            eta[3] = eta_ders[:, 0, 0]
            eta[4] = 0.

            e_1 = np.concatenate((eta[5][:, None, None],
                                  eta[6][:, None, None]),
                                  axis=1)
            e_2 = np.concatenate((eta[6][:, None, None],
                                  eta[7][:, None, None]),
                                  axis=1)
            eta_ders = np.concatenate([e_1, e_2], axis=2)
            eta_ders = rotation.T @ eta_ders @ rotation
            eta[5] = eta_ders[:, 0, 0]
            eta[6] = 0.
            eta[7] = 0.

            spec = self.ray_geometry(eta, pos_rcr, compute_derivatives=True)

            spec['i_scale'] = self.surface.dx
            spec['amp_scale'] = 1 / np.sqrt(np.abs(spec['hessian'][:, 1, 1]))
            spec["pulse_premult"] = "exp(-3j * pi / 4) * sqrt(f_a / c)"
            yield spec


    def aniso_KA_byangle(self, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """1D stationary phase scatter approximation with 1D varing surface"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            eta = anisotopic_igral(self.surface, pos_rcr, eta_interp,
                                   e_dx_interp, e_dy_interp,
                                   e_dxdx_interp, e_dydy_interp)
            spec = self.ray_geometry(eta, pos_rcr, compute_derivatives=True)

            spec['i_scale'] = self.surface.dx
            spec['amp_scale'] = 1 / np.sqrt(np.abs(spec['hessian'][:, 1, 1]))
            spec["pulse_premult"] = "exp(-3j * pi / 4) * sqrt(f_a / c)"
            yield spec


    def eigen_KA_byangle(self, eta, eta_interp, e_dx_interp, e_dy_interp,
                         e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
        """one bounce eigenray approximation of 2D surface scatter"""
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            points = stationary_points(self.surface, pos_rcr, eta, eta_interp,
                                       e_dx_interp, e_dy_interp, e_dxdx_interp,
                                       e_dxdy_interp, e_dydy_interp)
            spec = self.ray_geometry(points, pos_rcr, compute_derivatives=True)

            # stationary phase in 2D
            amp_scale = []
            premult = []
            for hess in spec['hessian']:
                vals = eigvals(hess)
                det = np.prod(vals)
                sig = np.sum(np.sign(vals))
                scale = 1 / np.sqrt(np.abs(det))
                amp_scale.append(scale)
                premult.append(f'-1j * exp(-1j * pi * {sig} / 4)')
            amp_scale = np.array(amp_scale)

            spec['sta_points'] = points[:2, :]
            spec['i_scale'] = 1.0
            spec['amp_scale'] = amp_scale
            spec["pulse_premult"] = premult
            yield spec


    def full_KA_byangle(self, eta):
        """Generate specifications of a scatter calculation"""
        # add x and y axis to eta to match stationary phase definitions
        eta_list = [np.broadcast_to(self.surface.x_a[:, None], eta[0].shape),
                    np.broadcast_to(self.surface.y_a[None, :], eta[0].shape)]
        eta_list += [e for e in eta]
        for th in self.xmission.theta:
            pos_rcr = self._pos_rcr(th)
            igral_mask = integral_mask(self.realization, th, self.xmission)
            spec = self.ray_geometry(eta_list, pos_rcr,
                                      surface_mask=igral_mask)
            spec['i_scale'] = self.surface.dx ** 2
            spec["pulse_premult"] = "-1j * (f_a / c)"
            yield spec
