import numpy as np
from scipy.signal.windows import hanning

def pulse(fc, fs):
    """approximate mooring navigation pulses"""
    T = 0.009
    num_samples = np.ceil(T * fs)
    t_signal = np.arange(num_samples) / fs
    y_signal = np.sin(2 * np.pi * fc * t_signal) * hanning(t_signal.size)
    return t_signal, y_signal


