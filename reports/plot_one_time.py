import argparse
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from os import listdir
from os.path import join

parser = argparse.ArgumentParser(
                    prog = 'kasea',
                    description = 'executes experiment specified in toml file')
parser.add_argument('filename')

args = parser.parse_args()

post_pos = args.filename.split('/')
post_pos = post_pos[-1] if len(post_pos[-1]) else post_pos[-2]
post_pos = post_pos.split('.')[0]

data_dir = join('results', post_pos)

files = listdir(data_dir)
files.sort()
files = [join(data_dir, f) for f in files]

file_name = files[0]

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

ping = objectview(dict(np.load(file_name)))

plot_i = 1

def normalize(p_in, r_img):
    p_dB = 20 * np.log10(np.abs(hilbert(p_in + np.spacing(1), axis=-1)))
    p_ref = 20 * np.log10(4 * pi * r_img)
    return p_dB + p_ref

fig, ax = plt.subplots()
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_2D, ping.r_img)[plot_i], color='k')
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_iso, ping.r_img)[plot_i])
ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_ani, ping.r_img)[plot_i])
#ax.plot(1e3 * ping.t_a, normalize(ping.p_sca_eig, ping.r_img)[plot_i])
ax.set_ylim(-60, 5)

plt.show()
