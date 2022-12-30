import numpy as np
import numexpr as ne
from math import pi
from ldis import ldis_deepwater

g = 9.81
km = 370  # wavenumber at GC wave phase speed minimum

def directional_spectrum(delta, k, k_bearing, omni_spectrum):
    """calculate spreading function from delta"""

    spreading = ne.evaluate("(1 + delta * cos(2 * k_bearing)) / (2 * pi)")

    # multiply omni-directional spectrum with spreading function
    d_spec = ne.evaluate("omni_spectrum * spreading / k")

    d_spec[np.isnan(d_spec)] = 0
    d_spec[np.isinf(d_spec)] = 0
    return d_spec


# directional spectrum formulations
def e_delta(k, U20):
    """Elfouhaily Delta function"""

    # define wave speeds
    omega = ldis_deepwater(k)
    cphase = ne.evaluate("omega / k")
    cpeak = 1.14 * U20
    cmin = 0.23  # m/s

    # compute wind stress following Wu 1991
    U10 = U20 / 1.026
    C10 = (0.8 + 0.065 * U10) * 1e-3
    u_star = np.sqrt(C10) * U10

    # Elfouhaily paramters
    a0 = np.log(2) / 4
    ap = 4
    am = 0.13 * u_star / cmin

    expr = "tanh(a0 + ap * (cphase / cpeak) ** 2.5" \
                 + "+ am * (cmin / cphase) ** 2.5)"

    return ne.evaluate(expr)
