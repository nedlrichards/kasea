import numpy as np
from math import sqrt, copysign
from scipy.optimize import newton

def bound_axes(z_src, z_rcr, dr, offset, max_dur, c=1500., theta=None):
    """return axes that have delay less than max_dur
    """
    tau_img = np.sqrt(dr ** 2 + (z_src + z_rcr) ** 2)
    tau_img /= c
    tau_lim = tau_img + max_dur
    x_img = z_src * dr / (z_src + z_rcr)

    z_src2 = (z_src - copysign(offset, z_src)) ** 2
    z_rcr2 = (z_rcr - copysign(offset, z_rcr)) ** 2

    # find x bounds
    rooter = lambda x: sqrt(x ** 2 + z_src2) / c \
                       + sqrt((x - dr) ** 2 + z_rcr2) / c \
                       - tau_lim

    x_start = newton(rooter, 0)
    x_end = newton(rooter, dr)

    # find y bounds
    eps = 1e-5
    r1 = lambda x, y: sqrt(x ** 2 + y ** 2 + z_src2) / c \
                       + sqrt((x - dr) ** 2 + y ** 2 + z_rcr2) / c \
                       - tau_lim

    # need to find where y_lim is at a max
    r2 = lambda x, y_init: (newton(lambda y: r1(x, y), y_init)
                            - newton(lambda y: r1(x + eps, y), y_init)) \
                            / eps

    x_ymax = newton(lambda x: r2(x, -dr), x_img, tol=eps)
    y_max = newton(lambda y: r1(x_ymax, y), dr)

    if theta is None:
        return (x_start, x_end), (-y_max, y_max)

    theta = np.array(theta, ndmin=1)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    points = np.array([[x_start, -y_max], [x_start, y_max],
                       [x_end, -y_max], [x_end, y_max]])

    rot_bounds = points @ R
    x_min, y_min =  np.min(rot_bounds, axis=(1, 2))
    x_max, y_max =  np.max(rot_bounds, axis=(1, 2))

    return (x_min, x_max), (y_min, y_max)
