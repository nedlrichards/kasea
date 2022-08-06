import numpy as np
import numexpr as ne
from tempfile import mkdtemp
import shutil
from os import path


class SurfMMAP:
    """Class to create and manipulate surfaces on disk"""

    def __init__(self, surface, chunksize=1e7):
        """setup surface and save files"""
        self.surface = surface
        self.chunksize = chunksize
        self.tmpdir = mkdtemp()

        # generate a surface realization
        self.real_file = path.join(self.tmpdir, 'realization.dat')
        realization = surface.gen_realization()
        real_mmap = np.memmap(self.real_file, dtype='float64',
                              mode='w+', shape=realization.shape)
        real_mmap = realization
        real_mmap.flush()

        # setup file to store surfaces
        if surface.y_a is None:
            self.ndshape = (3, surface.x_a.size)
        else:
            self.ndshape = (5, surface.x_a.size, surface.y_a.size)

        self.eta_file = path.join(self.tmpdir, 'realization.dat')


    def synthesize(self, time):
        """Generate realization of surface"""

        real_mmap = np.memmap(self.real_file, dtype='float64', mode='r')
        realization = np.array(real_mmap)

        fp = np.memmap(self.eta_file, dtype='float64',
                            mode='w+', shape=ndshape)
        surf = self.surf

        if surface.y_a is None:
            fp[0] = surf.x_a
            fp[1] = surf.surf.surface_synthesis(realization, time=time,
                                                     derivative=None)
            fp[2] = surf.surf.surface_synthesis(realization, time=time,
                                                     derivative='x')
        else:
            fp[0] = surf.x_a[:, None]
            fp[1] = surf.y_a[None, :]
            fp[2] = surf.surf.surface_synthesis(realization, time=time,
                                                derivative=None)
            fp[3] = surf.surf.surface_synthesis(realization, time=time,
                                                derivative='x')
            fp[4] = surf.surf.surface_synthesis(realization, time=time,
                                                derivative='y')
        fp.flush()


    def load_surface(self, dr, theta, tau_max):
        """compute a vector of all points bounded by traveltime"""
        fp = np.memmap(self.eta_file, dtype='float64', mode='r', shape=ndshape)
        num_vals = fp.size
        num_chunks = int(np.ceil(num_vals / chunksize))

        inds = self.ndshape[1] // num_chunks

        return fp[:, :inds]


    def __del__(self):
        """delete temporary directory"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
