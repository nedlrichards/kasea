import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from kasea import Ping
from scipy.signal import correlate
from scipy.interpolate import interp1d, RegularGridInterpolator
from kasea.surfaces import directional_spectrum, ldis_deepwater

plt.ion()

experiment = 'experiments/pm_surface.toml'

ping = Ping(experiment)

spectra = ping.surface.spec_2D

# make surface isentropic
kx = ping.surface.kx[:, None]
ky = ping.surface.ky[None, :]

ky_i = np.argmin(np.abs(ky))

fig, ax = plt.subplots()
ax.plot(ldis_deepwater(kx), ping.surface.spec_2D[:, ky_i])
ax.plot(ldis_deepwater(kx), ping.surface.spec_1D)

fig, ax = plt.subplots()
cm = ax.pcolormesh(kx[:, 0], ky[0, :], spectra.T)
fig.colorbar(cm)
ax.set_xlim(0, 0.5)
ax.set_ylim(-0.5, 0.5)

k = ne.evaluate("sqrt(kx ** 2 + ky ** 2)")
k_bearing = ne.evaluate("arctan2(ky, kx)")

omni_ier = interp1d(ping.surface.kx, ping.surface.spec_1D, bounds_error=False, fill_value=0.)
omni_spectrum = omni_ier(k)

delta = ping.surface.delta * 0.
spec_2D = directional_spectrum(delta, k, k_bearing, omni_spectrum)
ping.surface.spec_2D = spec_2D

fig, ax = plt.subplots()
cm = ax.pcolormesh(kx[:, 0], ky[0, :], ping.surface.spec_2D.T)
fig.colorbar(cm)
ax.set_xlim(0, 0.5)
ax.set_ylim(-0.5, 0.5)

1/0


theta = np.pi / 6

x_a = ping.surface.x_a
y_a = ping.surface.y_a

eta = ping.realization()

dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)

eta_ier = RegularGridInterpolator([x_a, y_a], eta[0], bounds_error=False)
r_sa = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + (eta[0] - ping.surface.z_src) ** 2)

def est_hess(f):
    f_x = np.diff(f, axis=0) / dx
    f_y = np.diff(f, axis=1) / dx
    f_xx = np.diff(f_x, axis=0) / dx
    f_xy = np.diff(f_x, axis=1) / dx
    f_yy = np.diff(f_y, axis=1) / dx
    return (np.array([f_x[:, :-1], f_y[:-1, :]]),
           np.array([[f_xx[:, :-2], f_xy[:-1, :-1]], [f_xy[:-1, :-1], f_yy[:-2, :]]]))

jac, hess = est_hess(eta[0])

test_i = 150

fit = correlate(jac[0, :, test_i], eta[1, :, test_i], 'valid') \
    / correlate(eta[1, :, test_i], eta[1, :, test_i], 'valid')
fit = correlate(jac[1, test_i, :], eta[2, test_i, :], 'valid') \
    / correlate(eta[2, test_i, :], eta[2, test_i, :], 'valid')

fig, ax = plt.subplots()
ax.plot(x_a[:-1], jac[0, :, test_i])
ax.plot(x_a, eta[1, :, test_i])

fig, ax = plt.subplots()
ax.plot(y_a[:-1], jac[1, test_i, :])
ax.plot(y_a, eta[2, test_i, :])

jac, hess = est_hess(r_sa)

dr_dx = (x_a[:, None] + (eta[0] - ping.surface.z_src) * eta[1]) / r_sa
dr_dy = (y_a[None, :] + (eta[0] - ping.surface.z_src) * eta[2]) / r_sa

fit = correlate(jac[0, :, test_i], dr_dx[:, test_i], 'valid') \
    / correlate(jac[0, :, test_i], dr_dx[:, test_i], 'valid')
fit = correlate(jac[1, test_i, :], dr_dy[test_i, :], 'valid') \
    / correlate(dr_dy[test_i, :], dr_dy[test_i, :], 'valid')


fig, ax = plt.subplots()
ax.plot(x_a[:-1], jac[0, :, test_i])
ax.plot(x_a, dr_dx[:, test_i])

fig, ax = plt.subplots()
ax.plot(y_a[:-1], jac[1, test_i, :])
ax.plot(y_a, dr_dy[test_i, :])


1/0
X, Y = np.meshgrid(x_a, y_a)
coords = np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1)
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
rotated_coords = coords @ R

vals = eta_ier(rotated_coords)

