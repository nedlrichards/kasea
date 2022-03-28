import numpy as np
import matplotlib.pyplot as plt

from src.specification import Broadcast
from src.surfaces import Surface
from src.helpers import bound_tau_ras

plt.ion()

exper_file = 'src/experiments/canope_setup.toml'
experiment = Broadcast(exper_file, 1.)

realization = experiment.surface.realization()
eta = experiment.surface.surface_synthesis(realization)
eta_x = experiment.surface.surface_synthesis(realization, derivative='x')
eta_y = experiment.surface.surface_synthesis(realization, derivative='y')

# split up surface for KA calculations
x_a = experiment.surface.x_a
y_a = experiment.surface.y_a

split_x = 10
split_y = 10

x_inds = np.array_split(np.arange(x_a.size), split_x)
y_inds = np.array_split(np.arange(y_a.size), split_y)

#for x_i in x_inds:
    #for y_i in y_inds:
        #x_sub = x_a[x_i]
        #y_sub = y_a[y_i]
        #eta_sub = eta[np.ix_(x_i, y_i)]
        #eta_x_sub = eta_x[np.ix_(x_i, y_i)]
        #eta_y_sub = eta_y[np.ix_(x_i, y_i)]
        #break
    #break

x_i = x_inds[5]
y_i = y_inds[5]
x_sub = x_a[x_i]
y_sub = y_a[y_i]
eta_sub = eta[np.ix_(x_i, y_i)]
eta_x_sub = eta_x[np.ix_(x_i, y_i)]
eta_y_sub = eta_y[np.ix_(x_i, y_i)]


#def source_vector(xaxis, yaxis, eta, eta_x, eta_y, src, rcr):
    #"""terms needed for normal derivative of greens function from source"""
src = experiment.src
rcr = experiment.rcr
x_a = x_sub
y_a = y_sub
eta = eta_sub
eta_x = eta_x_sub
eta_y = eta_y_sub
c = experiment.c
dx = experiment.surface.dx

x_a = np.broadcast_to(x_a[:, None], eta.shape)
y_a = np.broadcast_to(y_a[None, :], eta.shape)

x_src, y_src, z_src = src
x_rcr, y_rcr, z_rcr = rcr

n = np.array([-eta_x, -eta_y, np.ones_like(eta)])
a_vec = np.array([x_a, y_a, eta])

ras_vec = a_vec - src[:, None, None]
ras_norm = np.linalg.norm(ras_vec, axis=0)
rra_norm = np.linalg.norm(rcr[:, None, None] - a_vec, axis=0)
tau_total = (ras_norm + rra_norm) / c
tau_mask = tau_total < experiment.t_max

n = n[:, tau_mask]
ras_vec = ras_vec[:, tau_mask]
ras_norm = ras_norm[tau_mask]
rra_norm = rra_norm[tau_mask]
tau_total = tau_total[tau_mask]

projection = np.sum(n * ras_vec, axis=0) / ras_norm

import numexpr as ne
from math import pi

f_a = experiment.f_a[:, None]
projection = projection[None, :]
tau_total = tau_total[None, :]
pulse_FT = experiment.pulse_FT[:, None]

exper = "pulse_FT * projection * exp(2j * pi * f_a * tau_total) " \
      + "* dx ** 2 / (8 * pi ** 2 * c * tau_total)"
igrand = ne.evaluate(exper)
ka = np.sum(igrand, axis=-1)

