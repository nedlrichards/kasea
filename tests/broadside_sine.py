import numpy as np
from math import pi
import matplotlib.pyplot as plt

from src import XMitt

plt.ion()

xmitt = XMitt('tests/sine_1d.toml', num_sample_chunk=5e6)
specs = xmitt.setup()

ts_ka_1D = xmitt.ping_surface(specs)

xmitt = XMitt('tests/sine_2d.toml', num_sample_chunk=5e6)
specs = xmitt.setup()

ts_ka_2D = xmitt.ping_surface(specs)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka_1D)
ax.plot(xmitt.t_a, ts_ka_2D)
