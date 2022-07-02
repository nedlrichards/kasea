import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.signal import hilbert
import numexpr as ne
from src import ne_strs

from src import XMitt

plt.ion()

save_dir = 'results/tests'

xmitt = XMitt('tests/sine_1d_cycle.toml', num_sample_chunk=5e6, save_dir=save_dir)
t_a_wave, ts_ka_1D = xmitt()

xmitt = XMitt('tests/sine_2d_45_cycle.toml', num_sample_chunk=5e6, save_dir=save_dir)
t_a_wave, ts_ks_45 = xmitt()

xmitt = XMitt('tests/sine_2d_60_cycle.toml', num_sample_chunk=5e6, save_dir=save_dir)
t_a_wave, ts_ks_60 = xmitt()

xmitt = XMitt('tests/sine_2d_90_cycle.toml', num_sample_chunk=5e6, save_dir=save_dir)
t_a_wave, ts_ks_90 = xmitt()

c = xmitt.experiment.c
p_ref_2D = 1 / (4 * pi * c * xmitt.experiment.tau_img)

p_ref_1D = np.abs(hankel2(0, 2 * pi * xmitt.fc * xmitt.experiment.tau_img) / 4)
ts_1D = np.real(hilbert(ts_ka_1D, axis=-1) * np.exp(-3j * pi / 4))

ts_1D_dB = 20 * np.log10(np.abs(hilbert(ts_1D, axis=-1)))
ts_1D_dB -= 20 * np.log10(p_ref_1D)


ts_45_dB = 20 * np.log10(np.abs(hilbert(ts_ks_45, axis=-1)))
ts_45_dB -= 20 * np.log10(p_ref_2D)

ts_60_dB = 20 * np.log10(np.abs(hilbert(ts_ks_60, axis=-1)))
ts_60_dB -= 20 * np.log10(p_ref_2D)

ts_90_dB = 20 * np.log10(np.abs(hilbert(ts_ks_90, axis=-1)))
ts_90_dB -= 20 * np.log10(p_ref_2D)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.5, 3))
ax = axes[0, 0]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_1D_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)

ax = axes[0, 1]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_45_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)

ax = axes[1, 0]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_60_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)

ax = axes[1, 1]
cm = ax.pcolormesh(t_a_wave, xmitt.t_a, ts_90_dB.T, vmin=-40, vmax=5,
                   cmap=plt.cm.gray_r)


#fig.colorbar(cm)

