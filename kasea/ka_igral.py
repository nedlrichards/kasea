import numpy as np
import numexpr as ne
from math import pi
from scipy.interpolate import interp1d

def spec_igral(xmission, spec, pulse_premult="1"):
    """ka integration in time domain with interpolation"""
    pulse = xmission.pulse_FT
    f_a = xmission.f_a_pulse
    if isinstance(pulse_premult, list):
        pulse_FT = [ne.evaluate(pm + " * pulse") for pm in pulse_premult]
        pulse_FT = np.array(pulse_FT)
        import ipdb; ipdb.set_trace()
    else:
        pulse_FT = ne.evaluate(pulse_premult + " * pulse")
    pulse_td = np.fft.irfft(pulse_FT, 10 * (pulse_FT.size - 1))
    t_a_pulse = np.arange(pulse_td.size) / (pulse_td.size + 1) \
              * xmission.t_a_pulse[-1]
    pulse_ier = interp1d(t_a_pulse, pulse_td, kind=3, assume_sorted=True)

    #string to compute product of two greens functions
    c = xmission.c
    proj = spec['proj']
    m_as = spec['m_as']
    m_ra = spec['m_ra']

    dn_green_product_str = "proj / (4 * pi * c * m_as * m_ra)"
    scale = ne.evaluate(dn_green_product_str)

    # specification of ka integrand
    t_a = xmission.t_a
    dt = (t_a[-1] - t_a[0]) / (t_a.size - 1)

    # sort by tau and then chunck computation
    t_ref = xmission.tau_img
    sort_i = np.argsort(spec['tau_ras'], kind='stable')

    sorted_time = spec['tau_ras'][sort_i] - t_ref
    sorted_scale = scale[sort_i]
    amp_scale = spec["amp_scale"][sort_i] if 'amp_scale' in spec else None

    ka = np.zeros(t_a.shape)

    for i, t in enumerate(t_a):
        start_i = np.searchsorted(sorted_time, t - t_a_pulse[-1], side='right')
        sorted_time = sorted_time[start_i:]
        sorted_scale = sorted_scale[start_i:]
        if amp_scale is not None: amp_scale = amp_scale[start_i:]
        end_i = np.searchsorted(sorted_time, t, side='left')
        if end_i == 0:
            continue
        igrand = pulse_ier(sorted_time[:end_i] + t_a_pulse[-1] - t) \
                 * sorted_scale[:end_i]
        ka[i] = np.sum(amp_scale[:end_i] * igrand) if amp_scale is not None else igrand.sum()

    ka *= spec['i_scale']
    return ka


def spec_igral_FT(xmission, spec):
    """perform ka calculation over a single chunk"""

    # specification of ka integrand
    t_a = xmission.t_a
    t_a_pulse = xmission.t_a_pulse
    dt = (t_a[-1] - t_a[0]) / (t_a.size - 1)
    f_a = xmission.f_a_pulse[:, None]
    c = xmission.c
    pulse = xmission.pulse_FT[:, None]

    pulse_premult = spec.get("pulse_premult", "1")
    if isinstance(pulse_premult, list):
        pulse_FT = np.array([ne.evaluate(pm) for pm in pulse_premult])
        pulse_FT = np.array(pulse_FT)[None, :] * pulse
    else:
        pulse_FT = ne.evaluate(pulse_premult + " * pulse")

    dn_green_product_str = "amp_scale"
    dn_green_product_str += " * proj * exp(-2j * pi * f_a * (tau_ras - tau_shift))"
    dn_green_product_str += " / (4 * pi * m_as * m_ra)"

    ka = np.zeros(t_a.size + t_a_pulse.size, dtype=np.float64)

    # sort by tau and then chunck computation
    t_ref = xmission.tau_img + t_a[0]
    nss = np.asarray((spec['tau_ras'] - t_ref) / dt, dtype=np.int64)
    n_min = nss.min()
    if n_min < 0:
        raise(ValueError('earliest arrival before start of timeseries'))
    n_max = nss.max()

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

        if pulse_FT.shape[1] > 1:
            ka_FT = pulse_FT[:, chunk] * ne.evaluate(dn_green_product_str)
        else:
            ka_FT = pulse_FT * ne.evaluate(dn_green_product_str)

        surf_FT = np.sum(ka_FT, axis=-1)
        surf_ts = np.fft.irfft(surf_FT)
        ka[i: i + surf_ts.size] += surf_ts

    ka *= spec['i_scale']
    ka = ka[:-t_a_pulse.size]
    return ka
