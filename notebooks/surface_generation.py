import numpy as np
from math import pi
import matplotlib.pyplot as plt
from src import XMitt

plt.ion()

toml_file = 'notebooks/flat.toml'
#toml_file = 'notebooks/sine.toml'
xmitt = XMitt(toml_file)

broadcast = xmitt.broadcast
surface = xmitt.surface
realization = xmitt.realization

one_time = np.load(xmitt.one_time(0))

x_a = one_time['x_a']
y_a = one_time['y_a'] if 'y_a' in one_time else None
eta = one_time['eta']

if surface.y_a is None:
    fig, ax = plt.subplots()
    ax.plot(x_a, eta)
else:
    fig, ax = plt.subplots()
    ax.pcolormesh(x_a, y_a, eta.T, cmap=plt.cm.coolwarm)

for th in surface.theta:
    tau_bounds = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + (eta - surface.z_src) ** 2) \
            + np.sqrt((surface.dr * np.cos(th) - x_a[:, None]) ** 2
                    + (surface.dr * np.sin(th) - y_a[None, :]) ** 2
                         + (surface.z_rcr - eta) ** 2)

    tau_bounds /= broadcast.c
    X, Y = np.meshgrid(x_a, y_a)
    ax.contour(X, Y, (tau_bounds < broadcast.tau_max).T)

