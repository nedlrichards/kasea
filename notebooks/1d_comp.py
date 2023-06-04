from kasea import Ping
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

plt.ion()

ping = Ping('experiments/gaussian_surface.toml')
self = ping

time = 0.

#TODO: Make this call aware of solution type (maybe eigenrays?)
self.realization.synthesize(time)
eta = self.realization()

# interpolators and higher derivatives required for stationary phase
x_a = self.surface.x_a
y_a = self.surface.y_a

eta_interp = RegularGridInterpolator((x_a, y_a), eta[0],
                                        bounds_error=False)
e_dx_interp = RegularGridInterpolator((x_a, y_a), eta[1],
                                        bounds_error=False)
e_dy_interp = RegularGridInterpolator((x_a, y_a), eta[2],
                                        bounds_error=False)
e_dxdx_interp = RegularGridInterpolator((x_a, y_a), eta[3],
                                        bounds_error=False)
e_dxdy_interp = RegularGridInterpolator((x_a, y_a), eta[4],
                                        bounds_error=False)
e_dydy_interp = RegularGridInterpolator((x_a, y_a), eta[5],
                                            bounds_error=False)
iers = [eta_interp, e_dx_interp, e_dy_interp, e_dxdx_interp,
        e_dxdy_interp, e_dydy_interp]

specs_iso = [spec for spec in self.iso_KA_byangle(*iers)]
specs_ani = [spec for spec in self.aniso_KA_byangle(*iers)]

fig, ax = plt.subplots()
ax.plot(specs_iso[0]['m_as'], specs_iso[0]['hessian'][:, 1, 1])
ax.plot(specs_ani[0]['m_as'], specs_ani[0]['hessian'][:, 1, 1])

