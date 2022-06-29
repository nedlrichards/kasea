import numpy as np
import numexpr as ne
from math import pi

g = 9.81
km = 370  # wavenumber at GC wave phase speed minimum

def PM(axis_in, U20, is_omega=False):
    """Compute the Pierson-Moskowitz spectrum by radial frequency"""

    if is_omega:
        omega = np.array(axis_in, ndmin=1)
    else:
        k = np.array(axis_in, ndmin=1)
        # use deep-water dispersion to compute frequency axis
        omega = ldis_deepwater(k)
        dwdk = ldis_deepwater(k, derivative=True)

    # Pierson Moskowitz parameters
    beta = 0.74
    alpha = 8.1e-3

    expr = "exp(-beta * g ** 4 / (omega * U20) ** 4) * alpha * g ** 2 / omega ** 5"

    #pm = np.exp(-beta * g ** 4 / (omega * U20) ** 4)
    #pm *= alpha * g ** 2 / omega ** 5
    if not is_omega:
        #pm *= dwdk
        expr = 'dwdk * ' + expr

    pm = ne.evaluate(expr)

    pm[omega <= 0] = 0
    pm[np.isnan(pm)] = 0

    return pm


def directional_spectrum(delta, k, k_bearing, omni_spectrum, theta):
    """calculate spreading function from delta"""

    spreading = ne.evaluate("(1 + delta * cos(2 * k_bearing + theta)) / (2 * pi)")

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

    #delta = np.tanh(a0
                    #+ ap * (cphase / cpeak) ** 2.5
                    #+ am * (cmin / cphase) ** 2.5)
    expr = "tanh(a0 + ap * (cphase / cpeak) ** 2.5" \
                 + "+ am * (cmin / cphase) ** 2.5)"

    return ne.evaluate(expr)


def ldis_deepwater(k, derivative=False):
    """linear dispersion relationship assuming deep water"""
    #gc = ne.evaluate("(1 + (k / km) ** 2)")
    if derivative:
        expr = "g * (1 + 3 * (k / km) ** 2)" \
             + " / (2 * sqrt(g * k * (1 + (k / km) ** 2)))"
        #dwdk = g * (1 + 3 * (k / km) ** 2) \
                #/ (2 * np.sqrt(g * k * gc))
    else:
        expr = "sqrt(g * k * (1 + (k / km) ** 2))"
        #omega = np.sqrt(g * k * gc)

    return ne.evaluate(expr)
