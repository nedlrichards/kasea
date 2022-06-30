import numpy as np
from math import pi
import matplotlib.pyplot as plt
import numexpr as ne
from src import ne_strs

from src import XMitt
from scipy.special import hankel2

plt.ion()

xmitt = XMitt('tests/flat_1d.toml')
specs = xmitt.setup()

ts_ka = xmitt.ping_surface(specs)

# explicit computation
ka_str = ne_strs.dn_green_product(src_type=xmitt.src_type)

f_a = xmitt.f_a[None, :]
c = xmitt.experiment.c
x_a = xmitt.x_a
dx = xmitt.dx

A = xmitt.experiment.surface.spec_1D * np.sqrt(2)
kx = xmitt.experiment.surface.kx

eta = np.zeros_like(x_a)
e_dx = np.zeros_like(x_a)

m_as = np.sqrt(x_a ** 2 + (eta - xmitt.experiment.src[-1]) ** 2)
m_ra = np.sqrt((xmitt.experiment.rcr[0] - x_a) ** 2
                + (xmitt.experiment.rcr[-1] - eta) ** 2)
tau_ras = (m_as + m_ra) / c
tau_i = tau_ras < xmitt.tau_max

proj = ((eta - xmitt.experiment.src[-1]) - e_dx * x_a) / m_as

m_as = m_as[tau_i, None]
m_ra = m_ra[tau_i, None]
proj = proj[tau_i, None]
tau_ras = tau_ras[tau_i, None]

s_m_as = specs['m_as'][:, None]
s_m_ra = specs['m_ra'][:, None]
s_proj = specs['proj'][:, None]
s_tau_ras = specs['tau_ras'][:, None]

# zero pad pulse to match ts length
pulse_FT = np.fft.rfft(xmitt.experiment.pulse, n=xmitt.t_a.size)

tau_shift = specs['t_rcr_ref']

phase = np.exp(-2j * pi * f_a * (tau_ras - tau_shift))
igrand = proj * phase / (4 * pi * np.sqrt(m_as * m_ra))

ts_test = np.fft.irfft(np.sum(pulse_FT[None, :] * igrand, axis=0))
ts_test *= xmitt.dx

p_img_FT = -(1j / 4) * xmitt.experiment.pulse_FT \
         * hankel2(0, 2 * pi * xmitt.surf_f_a * xmitt.experiment.tau_img) \
         * np.exp(2j * pi * xmitt.surf_f_a * (xmitt.experiment.tau_img + xmitt.t_a[0]))
p_img_FT[0] = 0
p_img = np.fft.irfft(p_img_FT)

p_ref = np.abs(hankel2(0, 2 * pi * xmitt.fc * xmitt.experiment.tau_img) / 4)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka / p_ref)
ax.plot(xmitt.t_a, ts_test / p_ref)
ax.plot(xmitt.surf_t_a, p_img / p_ref)

ax.set_ylim(-1.2, 1.2)
ax.set_ylabel('Amplitude re image')
ax.set_xlabel('Delay re image (ms)')
