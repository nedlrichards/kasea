import numpy as np
from math import pi
import matplotlib.pyplot as plt

from src import XMitt
from scipy.special import hankel2

plt.ion()

xmitt = XMitt('tests/flat_1d.toml')
specs = xmitt.setup()
ts_ka = xmitt.ping_surface(specs)

p_img_FT = -(1j / 4) * xmitt.experiment.pulse_FT \
         * hankel2(0, 2 * pi * xmitt.f_a * xmitt.experiment.tau_img) \
         * np.exp(2j * pi * xmitt.f_a * (xmitt.experiment.tau_img - xmitt.t_a[0]))
p_img_FT[0] = 0
p_img = np.fft.irfft(p_img_FT)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka)
ax.plot(xmitt.t_a, p_img)