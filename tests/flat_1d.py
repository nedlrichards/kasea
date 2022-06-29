import numpy as np
from math import pi
import matplotlib.pyplot as plt

from src import XMitt
from scipy.special import hankel2

plt.ion()

xmitt = XMitt('tests/flat_1d.toml')
specs = xmitt.setup()

import numexpr as ne
from src import ne_strs

ka_str = ne_strs.dn_green_product(src_type=xmitt.src_type)

f_a = xmitt.surf_f_a[None, :]
i_scale = specs['i_scale']
c = xmitt.experiment.c

dx_as = specs['dx_as'][:, None]
dx_ra = specs['dx_ra'][:, None]
dz_as = specs['dz_as'][:, None]
dz_ra = specs['dz_ra'][:, None]
m_as = specs['m_as'][:, None]
m_ra = specs['m_ra'][:, None]
proj = specs['proj'][:, None]
tau_ras = specs['tau_ras'][:, None]

tau_shift = specs['t_rcr_ref'] + specs['num_samp_shift'] * xmitt.dt
tau_shift = tau_shift[:, None]

surf_ts = np.fft.irfft(xmitt.experiment.pulse_FT * ne.evaluate(ka_str), axis=-1)
num_surf = surf_ts.shape[-1]

nss = specs['num_samp_shift']
ts = xmitt.t_a.size

pos_inds = np.broadcast_to(np.arange(surf_ts.shape[0])[:, None], surf_ts.shape)
t_inds = nss[:, None] + np.arange(num_surf)[None, :]

pos_inds = pos_inds.flatten()
t_inds = t_inds.flatten()
is_in = t_inds < ts
pos_inds = pos_inds[is_in]
t_inds = t_inds[is_in]

ka = np.zeros((surf_ts.shape[0], ts), dtype=np.float64)
ka[pos_inds, t_inds] = surf_ts.flatten()[is_in]

ts_ka = xmitt.ping_surface(specs)
ts_test = np.sum(ka, axis=0) * xmitt.dx

p_img_FT = -(1j / 4) * xmitt.experiment.pulse_FT \
         * hankel2(0, 2 * pi * xmitt.surf_f_a * xmitt.experiment.tau_img) \
         * np.exp(2j * pi * xmitt.surf_f_a * (xmitt.experiment.tau_img + xmitt.t_a[0]))
p_img_FT[0] = 0
p_img = np.fft.irfft(p_img_FT)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka)
ax.plot(xmitt.t_a, ts_test)
ax.plot(xmitt.surf_t_a, p_img)
