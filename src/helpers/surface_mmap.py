import numpy as np
import numexpr as ne
from tempfile import mkdtemp
import shutil
from os import path

class SurfMMAP:
    """Class to create and manipulate surfaces on disk"""

    def __init__(self, surface, chunksize=1e5):
        """setup surface and save files"""
        self.surface = surface
        self.chunksize = chunksize
        self.tmpdir = mkdtemp()

        # generate a surface realization
        realization = surface.gen_realization()

        if realization is not None:
            self.real_file = path.join(self.tmpdir, 'realization.dat')
            real_mmap = np.memmap(self.real_file, dtype='float64',
                                mode='w+', shape=realization.shape)
            real_mmap = realization
            real_mmap.flush()
        else:
            self.real_file = None

        # setup file to store surfaces
        if surface.y_a is None:
            self.ndshape = (surface.x_a.size, 3)
        else:
            self.ndshape = (surface.x_a.size, surface.y_a.size, 5)

        self.eta_file = path.join(self.tmpdir, 'realization.dat')


    def synthesize(self, time):
        """Generate realization of surface"""

        if self.real_file is not None:
            real_mmap = np.memmap(self.real_file, dtype='float64', mode='r')
            realization = np.array(real_mmap)
        else:
            realization = None

        fp = np.memmap(self.eta_file, dtype='float64', mode='w+',
                       shape=self.ndshape)
        surf = self.surface

        if surf.y_a is None:
            fp[:, 0] = surf.x_a
            fp[:, 1] = surf.surface_synthesis(realization, time=time, derivative=None)
            fp[:, 2] = surf.surface_synthesis(realization, time=time,
                                                     derivative='x')
        else:
            fp[:, :, 0] = surf.x_a[:, None]
            fp[:, :, 1] = surf.y_a[None, :]
            fp[:, :, 2] = surf.surface_synthesis(realization, time=time,
                                                 derivative=None)
            fp[:, :, 3] = surf.surface_synthesis(realization, time=time,
                                                 derivative='x')
            fp[:, :, 4] = surf.surface_synthesis(realization, time=time,
                                                 derivative='y')
        fp.flush()


    def load_surface(self, theta):
        """compute a vector of all points bounded by traveltime"""
        fp = np.memmap(self.eta_file, dtype='float64', mode='r',
                       shape=self.ndshape)
        num_vals = fp.size
        num_chunks = int(np.ceil(num_vals / self.chunksize))

        inds = self.ndshape[0] // num_chunks
        start_i = inds * np.arange(num_chunks + 1)

        # add ones to position to make up for remainer
        rem = fp.shape[0] - start_i[-1]
        start_i[-rem:] += np.arange(rem) + 1

        # setup delay bound calculation
        z_src = self.surface.z_src
        z_rcr = self.surface.z_rcr
        dr = self.surface.dr
        c = self.surface.c
        tau_max = self.surface.tau_max


        vals = []

        for i, j in zip(start_i[:-1], start_i[1:]):
            tester = fp[i: j, :,  :]
            x_a = tester[:, :, 0]
            y_a = tester[:, :, 1]
            eta = tester[:, :, 2]

            r_src = np.sqrt(x_a ** 2 + y_a ** 2 + (eta - z_src) ** 2)
            r_rcr = np.sqrt((dr * np.cos(theta) - x_a) ** 2
                            + (dr * np.sin(theta) - y_a) ** 2
                            + (z_rcr - eta) ** 2)
            tau_bounds = (r_src + r_rcr) / c

            is_in = (tau_bounds <= tau_max)
            vals.append(tester[is_in, :])
        return np.concatenate(vals)


    def __del__(self):
        """delete temporary directory"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
