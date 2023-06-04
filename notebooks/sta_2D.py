from math import pi
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from shapely.geometry import LineString

from kasea import Ping

plt.ion()

experiment = 'experiments/sine.toml'
ping = Ping(experiment)

self = ping

self.realization.synthesize(0.)
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


pos_rcr = self._pos_rcr(0.)
z = eta[0]
z_dx = eta[1]
z_dy = eta[2]

z_s = self.surface.z_src
x_r, y_r, z_r = pos_rcr

b_x_a = self.surface.x_a[:, None]
b_y_a = self.surface.y_a[None, :]

r_s_s = "sqrt(b_x_a ** 2 + b_y_a ** 2 + (z - z_s) ** 2)"
r_r_s = "sqrt((x_r - b_x_a) ** 2 + (y_r - b_y_a) ** 2 + (z_r - z) ** 2)"

r_src = ne.evaluate(r_s_s)
r_rcr = ne.evaluate(r_r_s)

fig, ax = plt.subplots()
ax.plot((r_src + r_rcr)[:, np.argmin(np.abs(y_a))])

df_x_s = "(b_x_a + (z - z_s) * z_dx) / r_src \
        - ((x_r - b_x_a) + (z_r - z) * z_dx) / r_rcr"
df_y_s = "(b_y_a + (z - z_s) * z_dy) / r_src \
        - ((y_r - b_y_a) + (z_r - z) * z_dy) / r_rcr"

dfdx = ne.evaluate(df_x_s)
dfdy = ne.evaluate(df_y_s)

fig, ax = plt.subplots()
ax.plot(dfdx[:, np.argmin(np.abs(y_a))])
ax.plot(np.diff((r_src + r_rcr)[:, np.argmin(np.abs(y_a))]))
ax.grid()

# contours are defined in sample index
dx_cntrs = measure.find_contours(dfdx, 0.)
dy_cntrs = measure.find_contours(dfdy, 0.)

fig, ax = plt.subplots()
for cnt in dx_cntrs:
    ax.plot(*cnt.T, color='r')
for cnt in dy_cntrs:
    ax.plot(*cnt.T, color='b')


