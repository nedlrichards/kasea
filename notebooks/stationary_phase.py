import numpy as np
from math import pi
import numexpr as ne
import matplotlib.pyplot as plt
from kasea import Ping

plt.ion()

toml_file = 'experiments/flat_surface.toml'

ping = Ping(toml_file)

frozen = Realization(surf, include_hessian=True)
frozen.synthesize(0.)

# interpolation for memory maps
x_a = frozen.x_a
y_a = frozen.y_a
z_s = xmitt.z_src
z_r = xmitt.z_rcr
dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)
fp = np.memmap(frozen.eta_file, dtype='float64', mode='r', shape=frozen.ndshape)

eta = fp[0].copy()
e_dx = fp[1].copy()
e_dy = fp[2].copy()
e_dxdx = fp[3].copy()
e_dxdy = fp[4].copy()
e_dydy = fp[5].copy()

bounds = np.array([[x_a[0], y_a[0]], [x_a[-1], y_a[-1]]])

sample_points = []

# isentrophic stationary phase

for th in np.deg2rad(xmitt.theta):

    r = bounds / np.array([[np.cos(th), np.sin(th)]])
    r_m_i = np.abs(r).argmin(axis=1)
    dr = np.diff(r[[0, 1], r_m_i])
    r_a = np.arange(np.ceil(dr / dx)) * dx
    r_a += (r[1, r_m_i[1]] - r_a[-1])
    x_th = np.cos(th) * r_a
    y_th = np.sin(th) * r_a
    sample_points.append(np.array([x_th, y_th]))

"""
fig, ax = plt.subplots()
for sp in sample_points:
    ax.plot(sp[0], sp[1])

ax.plot(np.full_like(y_a, x_a[-1]), y_a, 'k')
ax.plot(np.full_like(y_a, x_a[0]), y_a, 'k')
ax.plot(x_a, np.full_like(x_a, y_a[0]), 'k')
ax.plot(x_a, np.full_like(x_a, y_a[-1]), 'k')
"""

sample_points = []

# anisentrophic stationary phase
r_src = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + z_s ** 2)

for th in np.deg2rad(xmitt.theta):
    dr = xmitt.dr
    x_r = dr * np.cos(th)
    y_r = dr * np.sin(th)

    # Assume z=0
    r_rcr = np.sqrt((x_r - x_a[:, None]) ** 2 + (y_r - y_a[None, :]) ** 2 + z_r ** 2)
    r_total = r_src + r_rcr

    # compute derivative
    ds = y_a[None, :] / r_src - (y_r - y_a[None, :]) / r_rcr
    d2s = ((x_r - x_a[:, None]) ** 2 + z_r ** 2) / r_rcr ** 3 \
        + (x_a[:, None] ** 2 + z_s ** 2) / r_src ** 3

    # stationary phase position
    crossing = np.abs(np.diff(np.sign(ds)))
    closest_i = np.argmax(crossing, axis=1)
    is_in = np.any(crossing, axis=1)
    y0 = ds[np.arange(x_a.size), closest_i]
    y1 = ds[np.arange(x_a.size), closest_i + 1]
    slope = (y1 - y0) / dx
    delta_x = -y0 / slope
    zero_y = y_a[closest_i] + delta_x
    zero_y[~is_in] = np.nan
    sample_points.append(zero_y)

"""
fig, ax = plt.subplots()
for sp in sample_points:
    ax.plot(x_a, sp)
"""

# 2D stationary phase
th = np.deg2rad(90.)
dr = xmitt.dr
x_r = dr * np.cos(th)
y_r = dr * np.sin(th)

b_x_a = x_a[:, None]
b_y_a = y_a[None, :]

r_r_s = "sqrt((x_r - b_x_a) ** 2 + (y_r - b_y_a) ** 2 + (z_r - eta) ** 2)"
r_s_s = "sqrt(b_x_a ** 2 + b_y_a ** 2 + (eta - z_s) ** 2)"

df_x_s = "(b_x_a + (eta - z_s) * e_dx) / r_src \
          - ((x_r - b_x_a) + (z_r - eta) * e_dx) / r_rcr"
df_y_s = "(b_y_a + (eta - z_s) * e_dy) / r_src \
          - ((y_r - b_y_a) + (z_r - eta) * e_dy) / r_rcr"

# Amplitude calculations
# TODO: Sample surface at stationary points before this calculation
df_xx_s = """
((1  + e_dx ** 2 + e_dxdx * (eta - z_s)) * r_src
  - (b_x_a + (eta - z_s) * e_dx) ** 2 / r_src) / r_src ** 2
+ ((1 - e_dxdx * (z_r - eta) + e_dx ** 2) * r_rcr
   - ((x_r - b_x_a) + (z_r - eta) * e_dx) ** 2 / r_rcr) / r_rcr ** 2
"""

df_xy_s = """
((e_dxdy * (eta - z_s) + e_dx * e_dy) * r_src
 - (b_x_a + (eta - z_s) * e_dx) * (b_y_a + (eta - z_s) * e_dy) / r_src) / r_src ** 2
+ ((e_dx * e_dy - e_dxdy * (z_r - eta)) * r_rcr
   - ((x_r - b_x_a) + (z_r - eta) * e_dx) * ((y_r - b_y_a) + (z_r - eta) * e_dy) / r_rcr) / r_rcr ** 2
"""

df_yy_s = """
((1 + e_dy ** 2 + e_dydy * (eta - z_s)) * r_src
 - (b_y_a + (eta - z_s) * e_dy) ** 2 / r_src) / r_src ** 2
+ ((1 - e_dydy * (z_r - eta) + e_dy ** 2) * r_rcr
  - ((y_r - b_y_a) + (z_r - eta) * e_dy) ** 2 / r_rcr) / r_rcr ** 2
"""

r_rcr = ne.evaluate(''.join(r_r_s.split()))
r_src = ne.evaluate(''.join(r_s_s.split()))
f = r_src + r_rcr
dfdx = ne.evaluate(''.join(df_x_s.split()))
dfdy = ne.evaluate(''.join(df_y_s.split()))
dfdxx = ne.evaluate(''.join(df_xx_s.split()))
dfdxy = ne.evaluate(''.join(df_xy_s.split()))
dfdyy = ne.evaluate(''.join(df_yy_s.split()))

diff_f_x = np.diff(f, axis=0) / dx
diff_f_y = np.diff(f, axis=1) / dx
diff_f_xx = np.diff(f, n=2, axis=0) / dx ** 2
diff_f_xy = np.diff(np.diff(f, axis=0), axis=1)/ dx ** 2
diff_f_yy = np.diff(f, n=2, axis=1) / dx ** 2

fig, ax = plt.subplots(3, 2, figsize=(5, 7))
ax[0, 0].plot(x_a[:-1], diff_f_x[:, 30])
ax[0, 0].plot(x_a, dfdx[:, 30])

ax[1, 0].plot(y_a[:-1], diff_f_y[30, :])
ax[1, 0].plot(y_a, dfdy[30, :])


ax[0, 1].plot(x_a[:-2], diff_f_xx[:, 30])
ax[0, 1].plot(x_a, dfdxx[:, 30])

ax[1, 1].plot(x_a[:-1], diff_f_xy[:, 30])
ax[1, 1].plot(x_a, dfdxy[:, 30])

ax[2, 1].plot(y_a[:-2], diff_f_yy[30, :])
ax[2, 1].plot(y_a, dfdyy[30, :])


"""

dx_cntrs = measure.find_contours(dfdx, 0.)
dy_cntrs = measure.find_contours(dfdy, 0.)

dx_lines = []
for cnt in dx_cntrs:
    dx_lines.append(LineString(cnt * dx + np.array([x_a[0], y_a[0]])))

stationary_points = []
for cnt in dy_cntrs:
    line = LineString(cnt * dx + np.array([x_a[0], y_a[0]]))
    for dx_line in dx_lines:
        stationary_points.append(line.intersection(dx_line))

sp_coords = []
for sp in stationary_points:
    sp_coords.append([sp.x, sp.y])
sp_coords = np.array(sp_coords)

fig, ax = plt.subplots()
cm = ax.pcolormesh(x_a, y_a, dfdx.T, cmap=plt.cm.coolwarm, vmax=2, vmin=-2)
for cnt in dx_cntrs:
    ax.plot(cnt[:, 0] * dx + x_a[0], cnt[:, 1] * dx + x_a[1], 'k')
fig.colorbar(cm)

fig, ax = plt.subplots()
cm = ax.pcolormesh(x_a, y_a, dfdy.T, cmap=plt.cm.coolwarm, vmax=1, vmin=-1)
for cnt in dy_cntrs:
    ax.plot(cnt[:, 0] * dx + x_a[0], cnt[:, 1] * dx + x_a[1], 'k')
fig.colorbar(cm)
"""


