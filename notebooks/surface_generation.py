import numpy as np
import matplotlib.pyplot as plt
from src.surfaces import Surface

plt.ion()

dx = 0.1
kmax = 2 * np.pi / dx
xbounds = (0, 200)
ybounds = (-100, 100)

U20 = 10.

surf = Surface(xbounds, ybounds, kmax, U20)
realization = surf.realization()
eta = surf.surface_synthesis(realization)

fig, ax = plt.subplots()
ax.pcolormesh(surf.x_a, surf.y_a, eta, cmap=plt.cm.coolwarm)
