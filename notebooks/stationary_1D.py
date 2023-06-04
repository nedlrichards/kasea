import numpy as np
from math import pi
import numexpr as ne
import matplotlib.pyplot as plt
from kasea import Ping

plt.ion()

toml_file = 'experiments/basic_geometry.toml'
ping = Ping(toml_file)
out_file = ping.one_time(0)

results = np.load(out_file)

fig, ax = plt.subplots()
cm = ax.pcolormesh(results['x_a'], results['y_a'], results['eta'].T, vmax=1, vmin=-1)
fig.colorbar(cm)

ax.plot(results['iso_x_a'], results['iso_y_a'], 'k')
ax.plot(results['aniso_x_a'], results['aniso_y_a'], 'k')

ax.set_xlim(results['x_a'][0], results['x_a'][-1])
ax.set_ylim(results['y_a'][0], results['y_a'][-1])
