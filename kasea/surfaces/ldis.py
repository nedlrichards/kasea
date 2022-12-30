import numpy as np
import numexpr as ne
from math import pi

g = 9.81
km = 370  # wavenumber at GC wave phase speed minimum

def ldis_deepwater(k, derivative=False):
    """linear dispersion relationship assuming deep water"""
    #gc = ne.evaluate("(1 + (k / km) ** 2)")
    if derivative:
        expr = "g * (1 + 3 * (k / km) ** 2)" \
             + " / (2 * sqrt(g * k * (1 + (k / km) ** 2)))"
    else:
        expr = "sqrt(g * k * (1 + (k / km) ** 2))"

    return ne.evaluate(expr)
