import numpy as np
import matplotlib.pyplot as plt

plt.ion()

z1 = -15
z2 = -10
x_diff = 200

dx = 0.1
x_bounds = (-50, 250)
y_bounds = (-50, 50)

x_a = np.arange(x_bounds[0], x_bounds[1], dx)
y_a = np.arange(y_bounds[0], y_bounds[1], dx)

tau_min = np.sqrt(x_diff ** 2 + (z1 + z2) ** 2)
tau_max = tau_min + 10

tau = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2 + z1 ** 2) \
        + np.sqrt((x_diff - x_a[:, None]) ** 2 + y_a[None, :] ** 2 + z2 ** 2)

tau_2 = np.sqrt(x_a[:, None] ** 2 + y_a[None, :] ** 2) \
          + np.sqrt((x_diff - x_a[:, None]) ** 2 + y_a[None, :] ** 2) \


x1 = z1 * x_diff / (z1 + z2)
x2 = x_diff - x1

p1_min = np.sqrt(x1 ** 2 + z1 ** 2)
p2_min = np.sqrt(x2 ** 2 + z2 ** 2)

fig, ax = plt.subplots()
ax.plot(x_a, tau[:, 500] - tau_2[:, 500])
ax.plot(x_a, tau[:, 200] - tau_2[:, 200])
