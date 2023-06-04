import argparse
import numpy as np
from math import pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from os import listdir, makedirs
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
frame_dir = 'results/surface_frames/'

# save frames in run specific directory
save_dir = join('results/surface_frames/', post_pos)
makedirs(save_dir, exist_ok=True)

files = listdir(data_dir)
files.sort()
files = [join(data_dir, f) for f in files]

time = []
eta_max = 0
eta_min = 0
for f in files:
    file = np.load(f)
    eta = file['eta']
    time.append(file['time'])
    eta_max = max(eta_max, eta.max())
    eta_min = max(eta_min, eta.min())

time = np.array(time)

def plot_surface(fname):
    file = np.load(fname)
    x_a = file['x_a']
    y_a = file['y_a']
    eta = file['eta']

    fig, ax = plt.subplots(figsize=(6, 4))

    cm = ax.pcolormesh(x_a, y_a, eta.T, cmap=plt.cm.gray_r, vmax=eta_max, vmin=eta_min, rasterized=True)

    cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    fig.colorbar(cm, cax=cax)
    fig.supxlabel('time (s)')
    fig.supylabel('delay re image (ms)')


    return fig, ax

def add_sta_points(fname, index, ax):
    file = np.load(fname)
    sta_points = file[f'sta_points_{index:03}']
    ax.plot(sta_points[0], sta_points[1], 'o', color=f'C{index}')

for f in files:
    fig, ax = plot_surface(f)
    add_sta_points(f, 0, ax)
    add_sta_points(f, 1, ax)
    ax.plot([0], [0], marker='*', color='w', markeredgecolor='k', markersize=12)
    file = np.load(f)
    for i, _ in enumerate(file['theta']):
        pos = file[f'pos_rcr_{i:03}']
        ax.plot(pos[0], pos[1], marker='d', color='w', markeredgecolor='k', markersize=8)

    save_name = f.split('/')[-1].split('.')[0]
    fig.savefig(join(save_dir, save_name))
    plt.close(fig)
