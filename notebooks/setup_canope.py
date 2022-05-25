import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from src import XMitt

plt.ion()

exper_file = 'src/experiments/canope_setup.toml'
xmitt = XMitt(exper_file, 1.)

p_sca = []
#for i in range(3):
xmitt.generate_realization()
xmitt.surface_realization()
p_sca.append(xmitt.ping_surface())

p_sca = np.array(p_sca)

fig, ax = plt.subplots()
ax.plot(xmitt.t_a, 20 * np.log10(np.abs(hilbert(p_sca))).T)
ax.grid()

