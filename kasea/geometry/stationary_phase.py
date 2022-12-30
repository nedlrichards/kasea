import numpy as np
import numexpr as ne

def anisotopic_igral(theta, eta_interp, e_dx_interp, e_dy_interp,
                     e_dxdx_interp, e_dydy_interp):
    """integral after stationary phase along surface wavefront"""

    eta_aniso = []

    for th in xmitt.surface.theta:
        dr = xmitt.surface.dr
        x_r = dr * np.cos(th)
        y_r = dr * np.sin(th)
        z_src = xmitt.surface.z_src
        z_rcr = xmitt.surface.z_rcr

        x_a = xmitt.surface.x_a
        y_guess = np.sin(th) * xmitt.broadcast.x_img

        r_dy = lambda x, y: y / np.sqrt(x ** 2 + y ** 2 + z_src ** 2) \
                            + (y - y_r) / np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2 + z_rcr ** 2)
        r_d2y = lambda x, y: (x ** 2 + z_src ** 2) / np.sqrt(x ** 2 + y ** 2 + z_src ** 2) ** 3 \
                            + ((x - x_r) ** 2 + z_rcr ** 2) / np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2 + z_rcr ** 2) ** 3

        # interation of NR minimization
        y_0 = np.full_like(x_a, y_guess)
        dy_diff = 2

        while dy_diff > 0.5:
            y_1 = y_0 - r_dy(x_a, y_0) / r_d2y(x_a, y_0)
            dy_diff = np.max(np.abs(y_0 - y_1)) / xmitt.surface.dx
            y_0 = y_1

        sample_points = np.concatenate([x_a[:, None], y_1[:, None]], axis=1)

        eta_vals = eta_interp(sample_points)
        eta_der = e_dx_interp(sample_points)
        eta_2der = e_dxdx_interp(sample_points)

        eta_aniso.append([y_1, eta_vals, eta_der, eta_2der])

    return np.array(eta_aniso)

def isotopic_igral(theta, eta_interp, e_dx_interp, e_dy_interp,
                   e_dxdx_interp, e_dydy_interp):
    """integral after stationary phase perpedicular to straight line"""

    for th in xmitt.surface.theta:

        r = bounds / np.array([[np.cos(th), np.sin(th)]])
        r_m_i = np.abs(r).argmin(axis=1)
        dr = np.diff(r[[0, 1], r_m_i])
        r_a = np.arange(np.ceil(dr / dx)) * dx
        r_a += (r[1, r_m_i[1]] - r_a[-1])
        x_th = np.cos(th) * r_a
        y_th = np.sin(th) * r_a

        sample_points = np.concatenate([x_th[:, None], y_th[:, None]], axis=1)

        eta_vals = eta_interp(sample_points)
        eta_der = e_dx_interp(sample_points) * np.cos(th) \
            - e_dy_interp(sample_points) * np.sin(th)
        eta_2der = e_dxdx_interp(sample_points) * np.cos(th) \
            - e_dydy_interp(sample_points) * np.sin(th)

        eta_iso.append(np.array([r_a, eta_vals, eta_der, eta_2der]))


