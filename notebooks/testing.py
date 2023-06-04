import numpy as np
from math import pi
import matplotlib.pyplot as plt
import numexpr as ne

from src import XMitt


plt.ion()
toml_file = 'experiments/pm_rotated.toml'
xmitt = XMitt(toml_file)

surf = xmitt.surface()

"""
fd = np.diff(surf[0, comp_i, :]) / xmitt.dy

fig, ax = plt.subplots()
ax.plot(xmitt.y_a[1:], fd)
ax.plot(xmitt.y_a, surf[2, comp_i, :])

decimation = 100
p_surf = surf[0, ::decimation, ::decimation]

fig, ax = plt.subplots()
cm = ax.pcolormesh(xmitt.x_a[::decimation], xmitt.y_a[::decimation], p_surf.T, cmap=plt.cm.coolwarm)
fig.colorbar(cm)

"""

import numexpr as ne
from src import ne_strs

comp_i = 6000


surface_height = surf[0, :, comp_i]
surface_dx = surf[1, :, comp_i]

z_src = xmitt.z_src
z_rcr = xmitt.z_rcr

deg = np.deg2rad(0)
pos_src = -xmitt.x_img * np.array([np.cos(deg), np.sin(deg)])
pos_rcr = (xmitt.dr - xmitt.x_img) * np.array([np.cos(deg), np.sin(deg)])

(x_src, y_src) = pos_src
(x_rcr, y_rcr) = pos_rcr

# 1D distances
dx_as = xmitt.x_a - x_src
dx_ra = x_rcr - xmitt.x_a

i_scale = xmitt.dx

# isospeed delays to surface
dz_as = surface_height - z_src
dz_ra = z_rcr - surface_height

# compute src and receiver distances
m_as = ne.evaluate(ne_strs.m_as("2D"))
m_ra = ne.evaluate(ne_strs.m_ra("2D"))

# normal derivative projection
proj = ne.evaluate(ne_strs.proj(src_type="2D"))

# time axis
tau_img = xmitt.experiment.tau_img
tau_ras = (m_as + m_ra) / xmitt.experiment.c
# bound integration by delay time
tau_lim = xmitt.experiment.tau_max
tau_i = ne.evaluate("tau_ras < tau_lim")

# tau limit all arrays
tau_ras = tau_ras[tau_i]
t_rcr_ref = xmitt.tau_img + xmitt.t_a[0]
num_samp_shift = np.asarray((tau_ras - t_rcr_ref) / xmitt.dt,
                            dtype=np.int64)

dx_as = np.broadcast_to(dx_as, m_as.shape)[tau_i]
dx_ra = np.broadcast_to(dx_ra, m_as.shape)[tau_i]

dy_as = None
dy_ra = None

dz_as = dz_as[tau_i]
dz_ra = dz_ra[tau_i]

m_as = m_as[tau_i]
m_ra = m_ra[tau_i]

proj = proj[tau_i]

specs = {'t_rcr_ref':t_rcr_ref,
        'num_samp_shift':num_samp_shift,
        'i_scale':i_scale, 'tau_i':tau_i,
        'dx_as':dx_as, 'dy_as':dy_as, 'dz_as':dz_as,
        'dx_ra':dx_ra, 'dy_ra':dy_ra, 'dz_ra':dz_ra,
        'proj':proj, 'm_as':m_as, 'm_ra':m_ra, 'tau_ras':tau_ras,
        'src':pos_src, 'rcr':pos_rcr}

ka_test = xmitt.ping_surface(specs)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ka_test)

