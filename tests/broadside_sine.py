import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import hankel2
import numexpr as ne
from src import ne_strs

from src import XMitt

plt.ion()

xmitt = XMitt('tests/sine_1d.toml', num_sample_chunk=5e6)
specs = xmitt.setup()

ts_ka_1D = xmitt.ping_surface(specs)

# explicit computation
ka_str = ne_strs.dn_green_product(src_type=xmitt.src_type)

f_a = xmitt.f_a[None, :]
c = xmitt.experiment.c
x_a = xmitt.x_a
dx = xmitt.dx

A = xmitt.experiment.surface.spec_1D * np.sqrt(2)
kx = xmitt.experiment.surface.kx

eta = A * np.sin(kx * x_a)
e_dx = A * kx * np.cos(kx * x_a)

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
ts_test_1D = np.fft.irfft(np.sum(pulse_FT[None, :] * igrand, axis=0))
ts_test_1D *= xmitt.dx

xmitt = XMitt('tests/sine_2d.toml', num_sample_chunk=5e6)
specs = xmitt.setup()

ts_ka_2D = xmitt.ping_surface(specs)

p_ref_1D = np.abs(hankel2(0, 2 * pi * xmitt.fc * xmitt.experiment.tau_img) / 4)
p_ref_2D = 1 / (4 * pi * c * xmitt.tau_img)

igrand = np.exp(-1j * 3 * pi / 4) * np.sqrt(f_a / c) * proj * phase \
       / (4 * pi * np.sqrt((m_as + m_ra) * m_as * m_ra))
ts_test_sta = np.fft.irfft(np.sum(pulse_FT[None, :] * igrand, axis=0))
ts_test_sta *= xmitt.dx

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka_1D / p_ref_1D)
ax.plot(xmitt.t_a, ts_test_1D / p_ref_1D)
ax.plot(xmitt.t_a, ts_test_sta / p_ref_2D)
ax.plot(xmitt.t_a, ts_ka_2D / p_ref_2D)

