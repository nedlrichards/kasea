import numpy as np
from math import pi
import matplotlib.pyplot as plt
import numexpr as ne

from src import XMitt


plt.ion()
#toml_file = 'experiments/sine_rotated.toml'
toml_file = 'experiments/pm_rotated.toml'

xmitt = XMitt(toml_file)
for i in range(70):
    save_path = xmitt.one_time()


#surf = xmitt.surface()
#spec = next(xmitt._angle_gen(surf))
#p_sca = xmitt.ping_surface(spec)
