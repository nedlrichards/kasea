"""string to compute distance from source to surface"""
m_as = "sqrt(dx_as ** 2 + dz_as ** 2 + dy_as ** 2)"


"""string to compute distance from source to surface"""
m_ra = "sqrt(dx_ra ** 2 + dz_ra ** 2 + dy_ra ** 2)"


"""string to compute n dot grad"""
proj = "(-dx_as * surface_dx + dz_as - dy_as * surface_dy) / m_as"


"""string to compute product of two greens functions"""
dn_green_product = "proj * exp(-2j * pi * f_a * (tau_ras - tau_shift))"
dn_green_product +=  " * -1j * f_a / (4 * pi * c * m_as * m_ra)"
