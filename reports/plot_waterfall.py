from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from os import listdir
from os.path import join

plt.ion()

#file_name = 'results/sine_0_'
#dir_name = 'results/gaussian_surface'
dir_name = 'results/gaussian_kam11'
#dir_name = 'results/pm_surface'
#dir_name = 'results/pm_kam11'

files = listdir(dir_name)
files.sort()

#presure_types = ['p_sca_2D', 'p_sca_eig', 'p_sca_ani', 'p_sca_iso']
presure_types = ['p_sca_2D', 'p_sca_ani', 'p_sca_iso']

p_plot = []
time = []
for f in files:
    file = np.load(join(dir_name, f))
    press = []
    for pt in presure_types:
        press.append(file[pt])
    p_plot.append(np.array(press))
    time.append(file['time'])
p_plot = np.array(p_plot)
time = np.array(time)

p_dB = 20 * np.log10(np.abs(hilbert(p_plot, axis=-1)))
p_ref = 20 * np.log10(4 * pi * file['r_img'])
p_dB += p_ref
t_a = 1e3 * file['t_a']

def plot_one_theta(theta_i):

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,6))

    cm = axes[0, 0].pcolormesh(time, t_a, p_dB[:, 0, theta_i, :].T,
                               vmin=-40, vmax=5, cmap=plt.cm.gray_r, rasterized=True)
    cm = axes[0, 1].pcolormesh(time, t_a, p_dB[:, 1, theta_i, :].T,
                               vmin=-40, vmax=5, cmap=plt.cm.gray_r, rasterized=True)
    cm = axes[1, 0].pcolormesh(time, t_a, p_dB[:, 2, theta_i, :].T,
                               vmin=-40, vmax=5, cmap=plt.cm.gray_r, rasterized=True)
    #cm = axes[1, 1].pcolormesh(time, t_a, p_dB[:, 3, theta_i, :].T,
                               #vmin=-40, vmax=5, cmap=plt.cm.gray_r, rasterized=True)

    cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    fig.colorbar(cm, cax=cax)
    fig.supxlabel('time (s)')
    fig.supylabel('delay re image (ms)')
    # TODO: Last save didn't include theta
    #fig.suptitle(f'Propagation angle: {np.rad2deg(theta_i):.1f}$^\circ$')

    return fig, axes

[plot_one_theta(i) for i in range(p_plot.shape[2])]
