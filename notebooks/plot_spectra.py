import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from kasea import Ping

plt.ion()

pm_exp = 'experiments/pm_surface.toml'
gauss_exp = 'experiments/gaussian_surface.toml'

ping = Ping(gauss_exp)

def h_rms(surface):
    """definition of h_rms assumes a one sided 1D spectrum"""
    kx = surface.kx
    d_kx = (kx[-1] - kx[0]) / (kx.size - 1)
    h_rms_ana = surface.h_rms
    h_rms_1D = np.sum(surface.spec_1D) * d_kx
    h_rms_2D = 2 * np.sum(surface.spec_2D) * d_kx ** 2
    return np.array((h_rms_ana, h_rms_1D, h_rms_2D))

def sample_1D(surface):
    pm_1d = surface.spec_1D
    h_rms = []
    for i in range(20):
        samps = surface.rng.normal(size=2 * pm_1d.size)
        samps = samps.view(np.complex128)
        kmax = surface.kmax
        scaled = samps / np.sqrt(2)
        scaled[0] = np.real(samps[0])
        scaled[-1] = np.real(samps[-1])
        spec = np.sqrt(pm_1d * pm_1d.size * kmax) * scaled

        eta_1D = np.fft.irfft(spec)
        h_rms.append(np.sum(eta_1D ** 2) / eta_1D.size)
    return np.array(h_rms)

(h_rms_ana, h_rms_1D, h_rms_2D) = h_rms(ping.surface)
h_rms_1D_eta = sample_1D(ping.surface)

ping.realization.synthesize(0.)
eta = ping.realization()[0]
h_rms_2D_eta = np.sum(eta ** 2) / eta.size

fig, ax = plt.subplots()
ax.plot([0, 20], [h_rms_ana, h_rms_ana], 'k', linewidth=2)
ax.plot([0, 20], [h_rms_1D, h_rms_1D])
ax.plot([0, 20], [h_rms_2D, h_rms_2D])

ax.plot(h_rms_1D_eta)
ax.plot([0, 20], [h_rms_2D_eta, h_rms_2D_eta])

1/0




ping = Ping(gauss_exp)
gauss_kx = ping.surface.kx
gauss_1d = ping.surface.spec_1D
dk = (gauss_kx[-1] - gauss_kx[0]) / (gauss_kx.size - 1)
h_rms = np.sqrt(2 * np.sum(gauss_1d) * dk)

k = gauss_kx
l = ping.surface.corr_length
h = ping.surface.rms_height
test = (l * h ** 2 / (2 * np.sqrt(np.pi))) * np.exp(-k**2 * l ** 2 / 4)

fig, ax = plt.subplots()
ax.loglog(pm_kx, pm_1d)
ax.loglog(gauss_kx, gauss_1d)
ax.loglog(k, test)
ax.set_ylim(10e-8, 10e3)
ax.grid()

