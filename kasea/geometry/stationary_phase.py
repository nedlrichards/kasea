import numpy as np
import numexpr as ne
from skimage import measure
from shapely.geometry import LineString


def anisotopic_igral(surface, theta, eta_interp, e_dx_interp, e_dy_interp,
                     e_dxdx_interp, e_dydy_interp):
    """integral after stationary phase along surface wavefront"""

    dr = surface.dr
    x_r = dr * np.cos(theta)
    y_r = dr * np.sin(theta)
    z_src = surface.z_src
    z_rcr = surface.z_rcr

    x_a = surface.x_a
    y_a = surface.y_a
    y_guess = np.sin(theta) * surface.x_img

    r_dy = lambda x, y: y / np.sqrt(x ** 2 + y ** 2 + z_src ** 2) \
                        + (y - y_r) / np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2 + z_rcr ** 2)
    r_d2y = lambda x, y: (x ** 2 + z_src ** 2) / np.sqrt(x ** 2 + y ** 2 + z_src ** 2) ** 3 \
                        + ((x - x_r) ** 2 + z_rcr ** 2) / np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2 + z_rcr ** 2) ** 3

    # interation of NR minimization
    y_0 = np.full_like(x_a, y_guess)
    y_min = np.full_like(x_a, y_a[0])
    y_max = np.full_like(x_a, y_a[-1])

    dy_diff = 2

    while dy_diff > 0.1:
        tau_der = r_dy(x_a, y_0)

        # set new y bounds, force monotonic behavor of r_dy
        pos_i = tau_der > 0
        y_max[pos_i] = y_0[pos_i]
        neg_i = tau_der < 0
        y_min[neg_i] = y_0[neg_i]

        y_1 = y_0 - tau_der / r_d2y(x_a, y_0)

        min_i = y_1 < y_min
        max_i = y_1 > y_max

        # if value maxes out, split the difference
        y_1[min_i | max_i] = (y_min[min_i | max_i] + y_max[min_i | max_i]) / 2
        dy_diff = np.max(np.abs(y_0 - y_1)) / surface.dx

        y_0 = y_1

    sample_points = np.concatenate([x_a[:, None], y_1[:, None]], axis=1)

    eta_vals = eta_interp(sample_points)
    eta_der = e_dx_interp(sample_points)
    eta_2der = e_dxdx_interp(sample_points)
    results = np.array([x_a, y_1, eta_vals, eta_der, eta_2der])

    return results


def isotopic_igral(surface, theta, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dydy_interp):
    """integral after stationary phase perpedicular to straight line"""

    delta_x = surface.x_a[-1] - surface.x_a[0]
    delta_y = surface.y_a[-1] - surface.y_a[0]
    axis = surface.x_a if abs(delta_x) > abs(delta_y) else surface.y_a

    x_th = np.cos(theta) * axis
    y_th = np.sin(theta) * axis

    sample_points = np.concatenate([x_th[:, None], y_th[:, None]], axis=1)

    eta_vals = eta_interp(sample_points)
    eta_der = e_dx_interp(sample_points) * np.cos(theta) \
        - e_dy_interp(sample_points) * np.sin(theta)
    eta_2der = e_dxdx_interp(sample_points) * np.cos(theta) \
        - e_dydy_interp(sample_points) * np.sin(theta)

    results = np.array([x_th, y_th, eta_vals, eta_der, eta_2der])

    return results


def stationary_points(surface, theta, eta, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dxdy_interp, e_dydy_interp):
    """Stationary points of 2D integral
    eta is a 6xNxM array: z, dz_dx, dz_dy, dz2_dxdx, dz2_dxdy, dz2_dydy"""

    z = eta[0]
    z_dx = eta[1]
    z_dy = eta[2]
    z_s = surface.z_src
    x_r = surface.dr * np.cos(theta)
    y_r = surface.dr * np.sin(theta)
    z_r = surface.z_rcr

    b_x_a = surface.x_a[:, None]
    b_y_a = surface.y_a[None, :]

    r_s_s = "sqrt(b_x_a ** 2 + b_y_a ** 2 + (z - z_s) ** 2)"
    r_r_s = "sqrt((x_r - b_x_a) ** 2 + (y_r - b_y_a) ** 2 + (z_r - z) ** 2)"

    r_src = ne.evaluate(r_s_s)
    r_rcr = ne.evaluate(r_r_s)

    df_x_s = "(b_x_a + (z - z_s) * z_dx) / r_src \
            - ((x_r - b_x_a) + (z_r - z) * z_dx) / r_rcr"
    df_y_s = "(b_y_a + (z - z_s) * z_dy) / r_src \
            - ((y_r - b_y_a) + (z_r - z) * z_dy) / r_rcr"

    dfdx = ne.evaluate(df_x_s)
    dfdy = ne.evaluate(df_y_s)

    # contours are defined in sample index
    dx_cntrs = measure.find_contours(dfdx, 0.)
    dy_cntrs = measure.find_contours(dfdy, 0.)

    dx_lines = []
    for cnt in dx_cntrs:
        dx_lines.append(LineString(cnt))

    stationary_points = []
    for cnt in dy_cntrs:
        line = LineString(cnt)
        for dx_line in dx_lines:
            points = line.intersection(dx_line)
            stationary_points.append(np.array([points.x, points.y], ndmin=2))

    stationary_points = np.concatenate(stationary_points) * surface.dx \
                      + np.array([surface.x_a[0], surface.y_a[0]], ndmin=2)
    return stationary_points
