import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import hilbert
from src.experiments import CANOPE

fc = 4e3
fs = 10e3

plt.ion()

t_a, pulse = CANOPE.pulse(fc, fs)
t_a, pulse_off = CANOPE.pulse(fc - 500, fs)

N = 2**10
f_a = np.arange(N // 2 + 1) * fs / N
t_a = np.arange(pulse.size) / fs

auto = correlate(pulse, pulse)
off = correlate(pulse, pulse_off)
t_a_c = np.arange(auto.size) / fs

c_db = 20 * np.log10(np.abs(hilbert(auto)))
o_db = 20 * np.log10(np.abs(hilbert(off)))
p_db = 20 * np.log10(np.abs(hilbert(pulse)))
c_db -= np.max(c_db)
p_db -= np.max(p_db)
o_db -= np.max(o_db)

fig, ax = plt.subplots()
plt.plot(t_a_c, c_db)
plt.plot(t_a_c, o_db)
plt.plot(t_a + (auto.size / 4) / fs, p_db)

