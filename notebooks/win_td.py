import numpy as np
from scipy.signal.windows import blackmanharris, nuttall
import matplotlib.pyplot as plt

plt.ion()

fc = 2.5e3
f_cut = 1.9 * fc  # estimate of Q for pulse
fs = 2 * f_cut
T = 1 / fc
num_cycles = 5
total_time = num_cycles * T
num_samples = np.ceil(total_time * fs) + 1
t_signal = np.arange(num_samples) / fs
window = nuttall(t_signal.size)
y_signal = np.sin(2 * np.pi * fc * t_signal) * window

c = np.array([0.3635819, 0.4891775, 0.1365995, 0.0106411])
z = 2 * np.pi * np.arange(t_signal.size) / (t_signal.size - 1)
cust_win = c[0] - c[1] * np.cos(z) + c[2] * np.cos(2 * z) - c[3] * np.cos(3 * z)

fig, ax = plt.subplots()
ax.plot(t_signal, window, 'k')
ax.plot(t_signal, cust_win)
