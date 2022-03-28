import numpy as np
from scipy.special import hankel2

def greens(a_0, a_1, facous, c=1500., ndim=3):
    """Acoustic greens function in isotropic media"""
    k, d, nd = _setup(a_0, a_1, facous)

    if ndim == 2:
        p = 1j * hankel2(0, k * norm_diff) / 4
    elif ndim == 3:
        p = np.exp(-1j * k * norm_diff) / (4 * np.pi * norm_diff)
    else:
        raise(ValueError("ndim (number of dimensions) can be 2 or 3"))

    return p

def grad_greens(a_0, a_1, facous, c=1500., ndim=3):
    """Acoustic greens function in isotropic media"""
    k, d, nd = _setup(a_0, a_1, facous)

    if ndim == 2:
        p = 1j * hankel2(0, k * norm_diff) / 4
    elif ndim == 3:
        p = (-1j * k * d * np.exp(-1j * k * nd)) / (4 * np.pi * nd ** 2)
    else:
        raise(ValueError("ndim (number of dimensions) can be 2 or 3"))

    return p

def _setup(a_0, a_1, facous):
    """Commmon setup that deals with mixed input dimensionality"""
    k = 2 * np.pi * facous / c
    diff_a = a_1 - a_0
    norm_diff = np.linalg.norm(diff_a, axis=-1)
    return k, diff_a, norm_diff
