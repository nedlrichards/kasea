from math import pi
import numpy as np
import matplotlib.pyplot as plt

from kasea import Ping

plt.ion()

experiment = 'experiments/flat.toml'
ping = Ping(experiment)
save_path = ping.one_time(0.)

results = np.load(save_path)
t_a = results['t_a']

# choose angle of propagation
theta_i = 0

fig, ax = plt.subplots()
ax.plot(t_a, results['p_img'] * 4 * pi * results['r_img'], color='k', label='img')
ax.plot(t_a, results['p_sca_ani'][theta_i, :] * 4 * pi * results['r_img'], label='ani')
ax.plot(t_a, results['p_sca_iso'][theta_i, :] * 4 * pi * results['r_img'], label='iso')
ax.plot(t_a, results['p_sca_eig'][theta_i, :] * 4 * pi * results['r_img'], label='eig')
ax.plot(t_a, results['p_sca_2D'][theta_i, :] * 4 * pi * results['r_img'], label='eig')

ax.grid()
