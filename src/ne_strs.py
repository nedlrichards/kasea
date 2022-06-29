def m_as(src_type='3D'):
    """string to compute distance from source to surface"""
    mag_str = "dx_as ** 2 + dz_as ** 2"
    if src_type == '3D':
        mag_str += " + dy_as ** 2"
    mag_str = "sqrt(" + mag_str + ")"
    return mag_str


def m_ra(src_type='3D'):
    """string to compute distance from source to surface"""
    mag_str = "dx_ra ** 2 + dz_ra ** 2"
    if src_type == '3D':
        mag_str += " + dy_ra ** 2"
    mag_str = "sqrt(" + mag_str + ")"
    return mag_str


def proj(src_type='3D'):
    """string to compute n dot grad"""
    proj_str = "-dx_as * surface_dx + dz_as"
    if src_type == '3D':
        proj_str += " - dy_as * surface_dy"
    proj_str = "(" + proj_str + ") / m_as"
    return proj_str


def dn_green_product(src_type='3D'):
    """string to compute product of two greens functions"""
    phase = "exp(-2j * pi * f_a * (tau_ras - tau_shift))"
    if src_type == '3D':
        g_str = phase + " * -1j * f_a / (4 * pi * c * m_as * m_ra)"
    elif src_type == '2D':
        g_str = phase + " / (4 * pi * sqrt(m_as * m_ra))"
    else:
        raise(ValueError("Source type not implimented"))

    g_str = "proj * " + g_str
    return g_str
