import numpy as np
from math import pi
import matplotlib.pyplot as plt
from src import XMitt, Realization

from scipy import ndimage

plt.ion()

toml_file = 'notebooks/flat.toml'
xmitt = XMitt(toml_file)

surf = xmitt.surface
frozen = Realization(surf)
frozen.synthesize(0.)

# interpolation for memory maps
x_a = frozen.x_a
y_a = frozen.y_a
dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)
fp = np.memmap(frozen.eta_file, dtype='float64', mode='r', shape=frozen.ndshape)

bounds = np.array([[x_a[0], y_a[0]], [x_a[-1], y_a[-1]]])

sample_points = []

for th in np.deg2rad(xmitt.theta):

    r = bounds / np.array([[np.cos(th), np.sin(th)]])
    r_m_i = np.abs(r).argmin(axis=1)
    dr = np.diff(r[[0, 1], r_m_i])
    r_a = np.arange(np.ceil(dr / dx)) * dx
    r_a += (r[1, r_m_i[1]] - r_a[-1])
    x_th = np.cos(th) * r_a
    y_th = np.sin(th) * r_a
    sample_points.append(np.array([x_th, y_th]))


fig, ax = plt.subplots()
for sp in sample_points:
    ax.plot(sp[0], sp[1])

ax.plot(np.full_like(y_a, x_a[-1]), y_a, 'k')
ax.plot(np.full_like(y_a, x_a[0]), y_a, 'k')
ax.plot(x_a, np.full_like(x_a, y_a[0]), 'k')
ax.plot(x_a, np.full_like(x_a, y_a[-1]), 'k')

#tester = ndimage.map_coordinates(fp[0], [[10.4, 3.7], [56.8, 87.5]])

#delay_mask = frozen.integral_mask(xmitt.theta[0])
