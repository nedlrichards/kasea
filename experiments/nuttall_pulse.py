import numpy as np
from scipy.signal.windows import hanning

from scipy.signal.windows import blackmanharris, nuttall


def pulse(fc):
    """Q = 1 pulse"""
    fs = 1.9 * fc  # estimate of Q for pulse
    T = 1 / fc
    num_cycles = 5
    total_time = num_cycles * T
    num_samples = np.ceil(total_time * fs)
    t_signal = np.arange(num_samples) / fs
    y_signal = np.sin(2 * np.pi * fc * t_signal) * nuttall(t_signal.size)
    return t_signal, y_signal
