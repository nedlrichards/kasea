import numpy as np
import numexpr as ne

def stationary_phase(surface, theta):

    # interpolation for memory maps
    x_a = experiment.x_a
    y_a = experiment.y_a
    z_s = experiment.z_src
    z_r = experiment.z_rcr
    dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)

    eta = fp[0].copy()
    e_dx = fp[1].copy()
    e_dy = fp[2].copy()
    e_dxdx = fp[3].copy()
    e_dxdy = fp[4].copy()
    e_dydy = fp[5].copy()


