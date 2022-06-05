import numpy as np
from math import pi
import matplotlib.pyplot as plt

from src import XMitt

plt.ion()

xmitt = XMitt('tests/flat_2d.toml')
specs = xmitt.setup()
ts_ka = xmitt.ping_surface(specs)

c = xmitt.experiment.c
tau_img = xmitt.experiment.tau_img

p_img_FT = -(xmitt.experiment.pulse_FT / (4 * pi * c * tau_img)) \
         * np.exp(-2j * pi * xmitt.f_a * xmitt.t_a[0])
p_img_FT[0] = 0
p_img = np.fft.irfft(p_img_FT)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka)
ax.plot(xmitt.t_a, p_img)
