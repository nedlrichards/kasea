from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.ion()

file_name = 'results/sine_0_020.npz'
#file_name = 'results/flat_0_000.npz'

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

ping = objectview(dict(np.load(file_name)))


plot_i = 0

def normalize(p_in, r_img):
    p_dB = 20 * np.log10(np.abs(hilbert(p_in, axis=-1)))
    p_ref = 20 * np.log10(4 * pi * r_img)
    return p_dB + p_ref

fig, ax = plt.subplots()
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_2D, ping.r_img)[plot_i], color='k')
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_iso, ping.r_img)[plot_i])
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_ani, ping.r_img)[plot_i])
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_eig, ping.r_img)[plot_i])
ax.set_ylim(-60, 5)
