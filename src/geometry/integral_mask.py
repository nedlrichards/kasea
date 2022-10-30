import numpy as np
import numexpr as ne

def integral_mask(surface_realization, theta, broadcast, to_shadow=False):
    """Return surface positions within a total range of d_max"""

    # setup delay bound calculation
    z_s = broadcast.z_src
    z_r = broadcast.z_rcr
    dr = broadcast.dr
    d_max = broadcast.tau_max * broadcast.c

    z = np.array(surface_realization()[0])
    if surface_realization.y_a is None:
        x_a = surface_realization.x_a
        y_a = 0
        x_r = dr
        y_r = 0
    else:
        x_a = surface_realization.x_a[:, None]
        y_a = surface_realization.y_a[None, :]
        x_r = dr * np.cos(theta)
        y_r = dr * np.sin(theta)

    d_src = ne.evaluate("sqrt(x_a ** 2 + y_a ** 2 + (z - z_s) ** 2)")
    d_rcr = ne.evaluate("sqrt((x_r - x_a) ** 2 + (y_r - y_a) ** 2 + (z_r - z) ** 2)")
    distance_mask = (d_src + d_rcr) <= d_max
    if not to_shadow:
        return distance_mask

    # shadow test works on projected geometry
    proj_d_src = ne.evaluate("sqrt(x_a ** 2 + y_a ** 2)")
    proj_d_rcr = ne.evaluate("sqrt((x_r - x_a) ** 2 + (y_r - y_a) ** 2)")

    # restrict showning domain
    eta = z[distance_mask]
    d_src = d_src[distance_mask]
    d_rcr = d_rcr[distance_mask]
    proj_d_src = proj_d_src[distance_mask]
    proj_d_rcr = proj_d_rcr[distance_mask]

    # sort bearings by projected source and receiver distance
    sort_i_src = np.argsort(proj_d_src, kind='heapsort')
    sort_i_rcr = np.argsort(proj_d_rcr, kind='heapsort')

    # define same launch angle sign for source and receiver
    th_src = (eta - z_s) / d_src
    th_rcr = (eta - z_r) / d_rcr

    if surface_realization.y_a is None:
        src_shad_i = shadow_1d(proj_d_src[sort_i_src], th_src[sort_i_src])
        rcr_shad_i = shadow_1d(proj_d_rcr[sort_i_rcr], th_rcr[sort_i_rcr])
        return (src_shad_i | rcr_shad_i)

    proj_d_src = proj_d_src[sort_i_src]
    proj_d_rcr = proj_d_rcr[sort_i_rcr]

    th_src = th_src[sort_i_src]
    th_rcr = th_rcr[sort_i_rcr]

    phi_src = ne.evaluate("arctan2(y_a, x_a)")
    phi_rcr = ne.evaluate("arctan2(y_r - y_a, x_r - x_a)")
    phi_src[np.isnan(phi_src)] = 0
    phi_rcr[np.isnan(phi_rcr)] = 0
    phi_src = phi_src[distance_mask][sort_i_src]
    phi_rcr = phi_rcr[distance_mask][sort_i_rcr]

    return distance_mask, proj_d_src, th_src, phi_src, np.argsort(sort_i_src)

    src_shad_i = shadow_2d(proj_d_src[sort_i_src], phi_src[sort_i_src], th_src[sort_i_src])
    rcr_shad_i = shadow_2d(proj_d_rcr[sort_i_rcr], phi_rcr[sort_i_rcr], th_rcr[sort_i_rcr])
    return (src_shad_i | rcr_shad_i)


def shadow_1d(proj_r, theta):
    """Create mask where either source or receiver ray is shadowed"""
    pass


def _shadow_2d(proj_r, phi, theta):
    """Create mask where either source or receiver ray is shadowed"""
    pass


