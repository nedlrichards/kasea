from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import glob
plt.ion()

file_name = 'results/sine_0_'
#file_name = 'results/flat_0_000.npz'

files = glob.glob(file_name + '*.npz')
files.sort()

theta_i = 0
#pressure_type = 'p_sca_2D'
pressure_type = 'p_sca_eig'

p_plot = []
time = []
for f in files:
    file = np.load(f)
    p_plot.append(file[pressure_type][theta_i])
    time.append(file['time'])
p_plot = np.array(p_plot)
time = np.array(time)

p_dB = 20 * np.log10(np.abs(hilbert(p_plot, axis=-1)))
p_ref = 20 * np.log10(4 * pi * file['r_img'])
p_dB += p_ref

fig, ax = plt.subplots()
cm = plt.pcolormesh(time, 1e3 * file['t_a'], p_dB.T, vmin=-50, vmax=5)
fig.colorbar(cm)

