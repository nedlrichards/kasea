import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from kasea import Ping
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator, interp1d
from kasea.surfaces import directional_spectrum, Realization

plt.ion()

experiment = 'experiments/gaussian_surface.toml'

ping = Ping(experiment)
ping.realization.synthesize(0.)
eta = ping.realization()

theta = ping.xmission.theta[-1]

x_a = ping.surface.x_a
y_a = ping.surface.y_a

test_y_i = 800
r_as = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + (eta[0] - ping.surface.z_src) ** 2)
r_ra = np.sqrt((ping.surface.dr - x_a[:, None]) ** 2 + y_a[None, :] ** 2
               + (ping.surface.z_rcr - eta[0]) ** 2)

tau_ras = (r_as + r_ra) / ping.surface.c

dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)

# compute delay derivatives with ping module
pos_rcr = (ping.surface.dr, 0., ping.surface.z_rcr)
# different definition of eta used internally in ping
eta_list = [x_a, np.broadcast_to(ping.surface.y_a[test_y_i], x_a.shape)]
eta_list += [e[:, test_y_i] for e in eta]
specs = ping.ray_geometry(eta_list, pos_rcr, compute_derivatives=True)

hess_ana = specs['hessian']

def est_hess(f):
    """Finite difference estimates of derivatives hessian"""
    f_xx = (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx ** 2
    f_xy = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] +  f[:-2, :-2]) \
         / (4 * dx ** 2)
    f_yy = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx ** 2
    return np.array([[f_xx, f_xy], [f_xy, f_yy]])

hess = est_hess(tau_ras)

fig, ax = plt.subplots()
ax.plot(x_a, hess_ana[:, 0, 0] * 1e3)
ax.plot(x_a[1:-1], hess[0, 0, :, test_y_i] * 1e3)

fig, ax = plt.subplots()
ax.plot(x_a, hess_ana[:, 0, 1] * 1e3)
ax.plot(x_a[1:-1], hess[0, 1, :, test_y_i] * 1e3)

fig, ax = plt.subplots()
ax.plot(x_a, hess_ana[:, 1, 1] * 1e3)
ax.plot(x_a[1:-1], hess[1, 1, :, test_y_i] * 1e3)

# grid rotation test

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

test_i = -1
theta = ping.xmission.theta[test_i]
rotation = np.array(([np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]))
specs = [spec for spec in ping.iso_KA_byangle(*iers)]

tau_ras = specs[test_i]['tau_ras']
tau_xx = (tau_ras[2:] - 2 * tau_ras[1:-1] + tau_ras[:-2]) / dx ** 2

hess_ana = specs[test_i]['hessian']
rot_hess = rotation.T @ hess_ana @ rotation

fig, ax = plt.subplots()
ax.plot(x_a[1:-1], tau_xx)
ax.plot(x_a, rot_hess[:, 0, 0], 'k')

ax.plot(x_a, hess_ana[:, 0, 0])

