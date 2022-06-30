import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.signal import hilbert
import numexpr as ne
from src import ne_strs

from src import XMitt

plt.ion()

xmitt = XMitt('tests/sine_1d_cycle.toml', num_sample_chunk=5e6)
t_a_wave, ts_ka_1D = xmitt()

xmitt = XMitt('tests/sine_2d_cycle.toml', num_sample_chunk=5e6)
t_a_wave, ts_ka_2D = xmitt()

# add phase shift to 1D results
c = xmitt.experiment.c
p_ref_1D = np.abs(hankel2(0, 2 * pi * xmitt.fc * xmitt.experiment.tau_img) / 4)
p_ref_2D = 1 / (4 * pi * c * xmitt.experiment.tau_img)

ts_ka_1D = np.real(hilbert(ts_ka_1D, axis=-1) * np.exp(-3j * pi / 4))

ts_1D_dB = 20 * np.log10(np.abs(hilbert(ts_ka_1D, axis=-1)))
ts_1D_dB -= 20 * np.log10(p_ref_1D)

ts_2D_dB = 20 * np.log10(np.abs(hilbert(ts_ka_2D, axis=-1)))
ts_2D_dB -= 20 * np.log10(p_ref_2D)


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6.5, 3))
ax = axes[0]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_1D_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)

ax = axes[1]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_2D_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)

#fig.colorbar(cm)


