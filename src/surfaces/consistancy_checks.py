import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal.windows import blackmanharris
from math import pi

def wn_grid_to_uni(wave_number_axis, spectrum):
    """Integrate along radials to estimate uni-directional spectrum"""
    dkw = (wave_number_axis[-1] - wave_number_axis[0]) \
        / (wave_number_axis.size - 1)

    ky = np.hstack([-(wave_number_axis[1:-1])[::-1], wave_number_axis])
    ier = RectBivariateSpline(ky, wave_number_axis, spectrum)

    synth_rad = []
    for i, r in enumerate(wave_number_axis):
        if i == 0:
            synth_rad.append(0)
            continue

        num_t = int(np.ceil(pi * r / dkw))
        if num_t % 2 == 0: num_t += 1

        theta = np.linspace(-np.pi / 2, np.pi / 2, num_t)
        dt = pi / num_t
        xi = r * np.cos(theta)
        yi = r * np.sin(theta)
        f_int = ier(yi, xi, grid=False)
        tsum = 2 * np.pi * r * f_int.sum() / num_t
        synth_rad.append(tsum)

    return np.array(synth_rad)

def spatial_to_wn(eta_grid, kmax):
    """Synthesize spatial grid to estimate wavenumber spectrum"""
    win_len = eta_grid.shape[0]
    # synthesize wave number field
    win_2D = blackmanharris(win_len)[:, None] * blackmanharris(win_len)[None, :]
    s2 = np.sum(win_2D ** 2)

    # transform and scale result
    syn_kxky = np.abs(np.fft.rfft2(win_2D * eta_grid, axes=(1,0))) ** 2
    syn_kxky *= 2 / (kmax * s2)

    # the fft shift makes thinking about this easier
    syn_kxky = np.fft.fftshift(syn_kxky, axes=1)
    return syn_kxky


