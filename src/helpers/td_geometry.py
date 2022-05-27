import numpy as np
from math import sqrt, copysign
from scipy.optimize import newton

def bound_axes(src, rcr, dx, offset, max_dur, c=1500.):
    """return axes that have delay less than max_dur
    """
    if src.size == 3:
        x_src, y_src, z_src = src
        x_rcr, y_rcr, z_rcr = rcr
    else:
        x_src, z_src = src
        x_rcr, z_rcr = rcr
        y_src = None
        y_rcr = None

    tau_img = np.sqrt(x_rcr ** 2 + (z_src + z_rcr) ** 2)
    tau_img /= c
    tau_lim = tau_img + max_dur

    z_src2 = (z_src + copysign(offset, z_src)) ** 2
    z_rcr2 = (z_rcr + copysign(offset, z_rcr)) ** 2

    # find x bounds
    rooter = lambda x: sqrt(x ** 2 + z_src2) / c \
                       + sqrt((x - x_rcr) ** 2 + z_rcr2) / c \
                       - tau_lim

    x_start = newton(rooter, 0)
    x_end = newton(rooter, x_rcr)

    if y_src is None:
        return (x_start, x_end)

    # find y bounds
    x_img = z_src * x_rcr / (z_src + z_rcr)
    eps = 1e-5
    r1 = lambda x, y: sqrt(x ** 2 + y ** 2 + z_src2) / c \
                       + sqrt((x - x_rcr) ** 2 + y ** 2 + z_rcr2) / c \
                       - tau_lim

    # need to find where y_lim is at a max
    r2 = lambda x, y_init: (newton(lambda y: r1(x, y), y_init)
                            - newton(lambda y: r1(x + eps, y), y_init)) \
                            / eps

    x_ymax = newton(lambda x: r2(x, -x_rcr), x_img, tol=eps)
    y_max = newton(lambda y: r1(x_ymax, y), x_rcr)

    return (x_start, x_end), (-y_max, y_max)


def bound_tau_ras(x_a, y_a, eta, src, rcr, t_max):
    """Compute mask bounds where tau_ras <= t_max"""

    x_src, y_src, z_src = src
    x_rcr, y_rcr, z_rcr = rcr

    #TODO: assumes r_src and r_rcr have y=0
    if np.abs(x_src) > 1e-5 | np.abs(y_src) > 1e-5 | np.abs(y_rcr) > 1e-5:
        1/0

    tau_as = np.sqrt((x_src - x_a) ** 2 + (y_src - y_a) ** 2
                     + (z_src - eta) ** 2)
    tau_as /= c

    tau_ra = np.sqrt((x_rcr - x_a) ** 2 + (y_rcr - y_a) ** 2
                     + (z_rcr - eta) ** 2)
    tau_ra /= c

    tau_mask = (tau_as + tau_ra) <= t_max

    return tau_mask
