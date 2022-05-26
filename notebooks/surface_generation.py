import numpy as np
from math import pi
import matplotlib.pyplot as plt
from src.surfaces import Surface

plt.ion()

dx = 0.1
kmax = 2 * np.pi / dx
xbounds = (0, 100)
#ybounds = (-50, 50)
ybounds = None

surface_dict = {'type':'PM', 'U20':10., 'z_max':1.}
#surface_dict = {'type':'sine', 'H':1., 'L':40, 'theta':0, 'z_max':1.}

surf = Surface(xbounds, ybounds, kmax, surface_dict)
realization = surf.realization()
eta = surf.surface_synthesis(realization)

if ybounds is None:
    fig, ax = plt.subplots()
    ax.plot(surf.x_a, eta)
else:
    fig, ax = plt.subplots()
    ax.pcolormesh(surf.x_a, surf.y_a, eta.T, cmap=plt.cm.coolwarm)
