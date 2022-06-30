import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.signal import hilbert
import numexpr as ne
from src import ne_strs

from src import XMitt

plt.ion()

xmitt = XMitt('tests/sine_1d_cycle.toml', num_sample_chunk=5e6)
ts_ka_1D = xmitt()

#xmitt = XMitt('tests/sine_2d.toml', num_sample_chunk=5e6)
#ts_ka_2D = xmitt()

# add phase shift to 1D results
#ts_ka_1D = np.real(hilbert(ts_ka_1D) * np.exp(-3j * pi / 4))

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, ts_ka_1D / p_ref_1D)
#ax.plot(xmitt.t_a, ts_ka_2D / p_ref_2D)

