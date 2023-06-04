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

# grid rotation
theta = np.pi / 6

x_a = ping.surface.x_a
y_a = ping.surface.y_a

dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)

def est_hess(f):
    """Finite difference estimates of derivatives in jacobian and hessian"""
    f_x = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
    f_y = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
    f_xx = (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx ** 2
    f_xy = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] +  f[:-2, :-2]) \
         / (4 * dx ** 2)
    f_yy = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx ** 2
    return (np.array([f_x, f_y]),
           np.array([[f_xx, f_xy], [f_xy, f_yy]]))

jac, hess = est_hess(eta[0])

test_i = 150

fit = correlate(jac[0, :, test_i], eta[1, :, test_i], 'valid') \
    / correlate(eta[1, :, test_i], eta[1, :, test_i], 'valid')

fit = correlate(jac[1, test_i, :], eta[2, test_i, :], 'valid') \
    / correlate(eta[2, test_i, :], eta[2, test_i, :], 'valid')

fit = correlate(hess[0, 0, :, test_i], eta[3, :, test_i], 'valid') \
    / correlate(eta[3, :, test_i], eta[3, :, test_i], 'valid')

fit = correlate(hess[0, 1, :, test_i], eta[4, :, test_i], 'valid') \
    / correlate(eta[4, :, test_i], eta[4, :, test_i], 'valid')

fit = correlate(jac[1, test_i, :], eta[2, test_i, :], 'valid') \
    / correlate(eta[2, test_i, :], eta[2, test_i, :], 'valid')


fig, ax = plt.subplots()
ax.plot(x_a[1:-1], jac[0, :, test_i])
ax.plot(x_a, eta[1, :, test_i])

fig, ax = plt.subplots()
ax.plot(y_a[1:-1], jac[1, test_i, :])
ax.plot(y_a, eta[2, test_i, :])

fig, ax = plt.subplots()
ax.plot(x_a[1:-1], hess[0, 0, :, test_i])
ax.plot(x_a, eta[3, :, test_i])

fig, ax = plt.subplots()
ax.plot(x_a[1:-1], hess[0, 1, :, test_i])
ax.plot(x_a, eta[4, :, test_i])

fig, ax = plt.subplots()
ax.plot(y_a[1:-1], hess[1, 1, test_i, :])
ax.plot(y_a, eta[5, test_i, :])


1/0

r_sa = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + (eta[0] - ping.surface.z_src) ** 2)
jac, hess = est_hess(r_sa)

dz = (eta[0] - ping.surface.z_src)
dr_dx = (x_a[:, None] + dz * eta[1]) / r_sa
dr_dy = (y_a[None, :] + dz * eta[2]) / r_sa

dr_dxdx = ((1 + eta[3] * dz + eta[1] ** 2) - dr_dx ** 2) / r_sa
dr_dxdy = (eta[4] * dz + dr_dx * dr_dy) / r_sa
dr_dydy = ((1 + eta[5] * dz + eta[2] ** 2) - dr_dy ** 2) / r_sa

fit = correlate(jac[0, :, test_i], dr_dx[:, test_i], 'valid') \
    / correlate(dr_dx[:, test_i], dr_dx[:, test_i], 'valid')
fit = correlate(jac[1, test_i, :], dr_dy[test_i, :], 'valid') \
    / correlate(dr_dy[test_i, :], dr_dy[test_i, :], 'valid')

fig, ax = plt.subplots()
ax.plot(x_a[1:-1], hess[0, 0, :, test_i])
ax.plot(x_a, dr_dxdx[:, test_i])

fig, ax = plt.subplots()
ax.plot(y_a[1:-1], hess[1, 1, test_i, :])
ax.plot(y_a, dr_dydy[test_i, :])

fig, ax = plt.subplots()
ax.plot(y_a[1:-1], hess[0, 1, test_i - 1, :])
ax.plot(y_a, dr_dxdy[test_i, :])


X, Y = np.meshgrid(x_a, y_a)
coords = np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1)
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
rotated_coords = coords @ R

eta_ier = RegularGridInterpolator([x_a, y_a], eta[0], bounds_error=False)
vals = eta_ier(rotated_coords)


