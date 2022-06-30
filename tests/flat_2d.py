import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import hankel2

from src import XMitt

plt.ion()

xmitt = XMitt('tests/flat_1d.toml')
specs = xmitt.setup()

ts_ka_1D = xmitt.ping_surface(specs)

xmitt = XMitt('tests/flat_2d.toml', num_sample_chunk=5e6)
specs = xmitt.setup()

ts_ka_2D = xmitt.ping_surface(specs)

c = xmitt.experiment.c
tau_img = xmitt.experiment.tau_img

pulse_FT = np.fft.rfft(xmitt.experiment.pulse, xmitt.t_a.size)

p_img_FT = -(1j / 4) * pulse_FT \
         * hankel2(0, 2 * pi * xmitt.f_a * xmitt.experiment.tau_img) \
         * np.exp(-2j * pi * xmitt.f_a * -(xmitt.experiment.tau_img + xmitt.t_a[0]))

p_img_FT[0] = 0
p_img_1D = np.fft.irfft(p_img_FT)
#p_img_1D = np.fft.irfft(p_img_FT * np.exp(-3j * pi / 4))

p_img_FT = -(pulse_FT / (4 * pi * c * tau_img)) \
        * np.exp(-2j * pi * xmitt.f_a * -xmitt.t_a[0])
p_img_FT[0] = 0

p_img_2D = np.fft.irfft(p_img_FT)

p_ref_1D = np.abs(hankel2(0, 2 * pi * xmitt.fc * xmitt.experiment.tau_img) / 4)
p_ref_2D = 1 / (4 * pi * c * tau_img)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka_1D / p_ref_1D)
ax.plot(xmitt.t_a, p_img_1D / p_ref_1D)
ax.plot(xmitt.t_a, ts_ka_2D / p_ref_2D)
ax.plot(xmitt.t_a, p_img_2D / p_ref_2D)

ax.set_ylim(-1.2, 1.2)
ax.set_ylabel('Amplitude re image')
ax.set_xlabel('Delay re image (ms)')

