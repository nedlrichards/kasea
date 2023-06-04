import numpy as np
from math import pi
import matplotlib.pyplot as plt
import numexpr as ne
import os
from scipy.signal import hilbert

from src import XMitt


plt.ion()
load_path = 'results'
results = [r for r in os.listdir(load_path) if r[:2] == 'pm']
#results = [r for r in os.listdir(load_path) if r[:4] == 'sine']

results.sort(key=lambda s: int(s.split('.')[0].split('_')[-1]))

p_sca = []
wave_time = []
for res in results:
    npz = np.load(os.path.join(load_path, res))
    p_sca.append(np.array([npz['p_sca'], npz['p_sca_1d']])[:, :, None, :])
    wave_time.append(npz['time'])

wave_time = np.array(wave_time)
t_a = npz['t_a']

r_img = npz['r_img']
p_ref_2D = 1 / (4 * pi * r_img)

p_sca = hilbert(np.concatenate(p_sca, axis=-2), axis=-1)
p_sca_dB = 20 * (np.log10(np.abs(p_sca)) - np.log10(p_ref_2D))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.5, 3))
ax = axes[0, 0]
cm = ax.pcolormesh(wave_time, t_a, p_sca_dB[0, 0, :, :].T,
                   vmin=-40, vmax=5, cmap=plt.cm.gray_r)

ax = axes[0, 1]
cm = ax.pcolormesh(wave_time, t_a, p_sca_dB[0, 1, :, :].T,
                   vmin=-40, vmax=5, cmap=plt.cm.gray_r)

ax = axes[1, 0]
cm = ax.pcolormesh(wave_time, t_a, p_sca_dB[0, 2, :, :].T,
                   vmin=-40, vmax=5, cmap=plt.cm.gray_r)

ax = axes[1, 1]
cm = ax.pcolormesh(wave_time, t_a, p_sca_dB[0, 3, :, :].T,
                   vmin=-40, vmax=5, cmap=plt.cm.gray_r)


#fig.colorbar(cm)
