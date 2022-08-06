import numpy as np
from math import pi
import matplotlib.pyplot as plt
from src import XMitt
from src.surfaces import Surface

plt.ion()

toml_file = 'notebooks/sine.toml'
xmitt = XMitt(toml_file)

surf = xmitt.surface
realization = surf.gen_realization()
eta = surf.surface_synthesis(realization)

if xmitt.y_a is None:
    fig, ax = plt.subplots()
    ax.plot(surf.x_a, eta)
else:
    fig, ax = plt.subplots()
    ax.pcolormesh(surf.x_a, surf.y_a, eta.T, cmap=plt.cm.coolwarm)

x_a = surf.x_a[:, None]
y_a = surf.y_a[None, :]

theta = np.deg2rad(xmitt.theta)

for th in theta:
    tau_bounds = np.sqrt(x_a ** 2 + y_a ** 2 + (eta - xmitt.z_src) ** 2) \
               + np.sqrt((xmitt.dr * np.cos(th) - x_a) ** 2
                         + (xmitt.dr * np.sin(th) - y_a) ** 2
                         + (xmitt.z_rcr - eta) ** 2)

    tau_bounds /= xmitt.experiment.c
    ax.contour(surf.x_a, surf.y_a, (tau_bounds < xmitt.tau_max).T)

