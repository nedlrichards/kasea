import numpy as np
from src import bound_axes

z_src = -20
z_rcr = -10
dr = 200.
offset = 0.
max_dur = 0.001

# resolution used to check bounds estimate
dx = 0.1

tau_img = np.sqrt(dr ** 2 + (z_src + z_rcr) ** 2) / 1500.
tau = lambda x, y: (np.sqrt(x ** 2 + y ** 2 +  z_src ** 2) \
                   + np.sqrt((dr - x) ** 2 + y ** 2 + z_rcr ** 2)) / 1500.

x_bounds, y_bounds = bound_axes(z_src, z_rcr, dr, offset, max_dur)
x_a = np.arange(x_bounds[0], x_bounds[1], dx)
y_a = np.arange(y_bounds[0], y_bounds[1], dx)

tau_grid = tau(x_a[:, None], y_a[None, :]) - tau_img

print(np.min(tau_grid[0, :]))
print(np.min(tau_grid[-1, :]))
print(np.min(tau_grid[:, 0]))
print(np.min(tau_grid[:, -1]))

t_test = np.deg2rad(np.array([0., 15., 30., 45.]))
x_bounds, y_bounds = bound_axes(z_src, z_rcr, dr, offset, max_dur, theta=t_test)

R = np.array([[np.cos(t_test), -np.sin(t_test)],
              [np.sin(t_test), np.cos(t_test)]])

points = np.array([[x_bounds[0], y_bounds[0]],
                   [x_bounds[0], y_bounds[1]],
                   [x_bounds[1], y_bounds[0]],
                   [x_bounds[1], y_bounds[1]]])

rot_bounds = points @ R

x_min, y_min =  np.min(np.min(rot_bounds, axis=1), axis=1)
x_max, y_max =  np.max(np.max(rot_bounds, axis=1), axis=1)

import matplotlib.pyplot as plt
plt.ion()

fig, ax = plt.subplots()
ax.plot(rot_bounds[0], rot_bounds[1], '.')

ax.plot(x_min, y_min, '.', color='k')
ax.plot(x_max, y_min, '.', color='k')
ax.plot(x_min, y_max, '.', color='k')
ax.plot(x_max, y_max, '.', color='k')
