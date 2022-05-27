import numpy as np
from math import pi
import matplotlib.pyplot as plt

from src import XMitt

plt.ion()

xmitt = XMitt('tests/flat_1d.toml')
ts_ka = xmitt.ping_surface()

p_img_FT = -xmitt.experiment.pulse_FT
p_img_FT *= np.exp(-2j * pi * xmitt.f_a * -xmitt.t_a[0])
p_img_FT /= xmitt.experiment.tau_img * xmitt.experiment.c
p_img = np.fft.irfft(p_img_FT)

fig, ax = plt.subplots()
#ax.plot(xmitt.t_a, ts_ka)
ax.plot(xmitt.t_a, p_img)
