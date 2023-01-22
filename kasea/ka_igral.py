import numpy as np
import numexpr as ne
from math import pi
from scipy.interpolate import interp1d

def spec_igral(xmission, spec, pulse_premult="1"):
    """ka integration in time domain with interpolation"""
    pulse = xmission.pulse_FT
    pulse_FT = ne.evaluate(pulse_premult + " * pulse")
    pulse_td = np.fft.irfft(pulse_FT)

    #string to compute product of two greens functions
    # * -1j * f_a
    dn_green_product_str = "proj / (4 * pi * c * m_as * m_ra)"

    # specification of ka integrand
    t_a = xmission.t_a
    dt = (t_a[-1] - t_a[0]) / (t_a.size - 1)
    f_a = xmission.f_a_pulse[:, None]
    c = xmission.c

    # sort by tau and then chunck computation
    t_ref = xmission.tau_img + t_a[0]
    sort_i = np.argsort(spec['tau_ras'], kind='stable')
    # TODO: do we need to sort tau_ras?
    sorted_time = spec['tau_ras'][sort_i]

    nss = np.asarray((spec['tau_ras'] - t_ref) / dt, dtype=np.int64)
    n_min = nss.min()
    n_max = nss.max()

    for i in range(n_min, n_max):
        chunk = (nss == i)

        # setup KA for one sample delay
        proj = spec['proj'][chunk][None, :]
        tau_ras = spec['tau_ras'][chunk][None, :]
        n_vals = nss[chunk]
        D_tau = n_vals[None, :] * xmission.dt
        tau_shift = t_ref + D_tau
        m_as = spec['m_as'][chunk][None, :]
        m_ra = spec['m_ra'][chunk][None, :]

        ka_FT = ne.evaluate("pulse * " + dn_green_product_str)
        ka[i: i + surf_ts.size] += surf_ts

    ka *= spec['i_scale']
    ka = ka[:num_t_a]
    return ka


def spec_igral_FT(xmission, spec):
    """perform ka calculation over a single chunk"""
    #string to compute product of two greens functions
    dn_green_product_str = spec.get("pulse_premult", "1")
    dn_green_product_str += " * " + "amp_scale"
    dn_green_product_str += " * proj * exp(-2j * pi * f_a * (tau_ras - tau_shift))"
    dn_green_product_str += " / (4 * pi * m_as * m_ra)"

    # specification of ka integrand
    t_a = xmission.t_a
    t_a_pulse = xmission.t_a_pulse
    dt = (t_a[-1] - t_a[0]) / (t_a.size - 1)
    f_a = xmission.f_a_pulse[:, None]
    c = xmission.c

    ka = np.zeros(t_a.size + t_a_pulse.size, dtype=np.float64)

    # sort by tau and then chunck computation
    t_ref = xmission.tau_img + t_a[0]
    nss = np.asarray((spec['tau_ras'] - t_ref) / dt, dtype=np.int64)
    n_min = nss.min()
    n_max = nss.max()
    pulse = xmission.pulse_FT[:, None]

    for i in range(n_min, t_a.size - 1):
        chunk = (nss == i)
        if not chunk.any():
            continue

        # setup KA for one sample delay
        proj = spec['proj'][chunk][None, :]
        amp_scale = spec["amp_scale"][chunk][None, :] if 'amp_scale' in spec else 1
        tau_ras = spec['tau_ras'][chunk][None, :]
        n_vals = nss[chunk]
        D_tau = n_vals[None, :] * xmission.dt
        tau_shift = t_ref + D_tau
        m_as = spec['m_as'][chunk][None, :]
        m_ra = spec['m_ra'][chunk][None, :]

        ka_FT = ne.evaluate("pulse * " + dn_green_product_str)

        surf_FT = np.sum(ka_FT, axis=-1)
        surf_ts = np.fft.irfft(surf_FT)
        ka[i: i + surf_ts.size] += surf_ts

    ka *= spec['i_scale']
    ka = ka[:-t_a_pulse.size]
    return ka
