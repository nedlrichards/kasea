import numpy as np
import numexpr as ne
from tempfile import mkdtemp
import shutil
from os import path

class Realization:
    """Class to create and manipulate surfaces on disk"""

    def __init__(self, surface, chunksize=1e5):
        """setup surface and save files"""
        self.surface = surface
        self.chunksize = chunksize
        self.tmpdir = mkdtemp()
        self.theta = 0.

        # generate a surface realization
        realization = surface.gen_realization()

        if realization is not None:
            self.real_file = path.join(self.tmpdir, 'realization.dat')
            self.real_shape = realization.shape
            real_mmap = np.memmap(self.real_file, dtype='complex128',
                                mode='w+', shape=self.real_shape)
            real_mmap[:] = realization[:]
            real_mmap.flush()
        else:
            # used for surfaces that don't require a spectrum realization
            self.real_file = None

        # setup file to store surfaces
        self.ndshape = (6, surface.x_a.size, surface.y_a.size)
        self.eta_file = path.join(self.tmpdir, 'eta.dat')


    def __call__(self):
        """Load memmap of surface"""
        real_mmap = np.memmap(self.eta_file, dtype='float64', mode='r', shape=self.ndshape)
        return real_mmap


    def synthesize(self, time):
        """Generate realization of surface"""
        if self.real_file is not None:
            real_mmap = np.memmap(self.real_file, dtype='complex128', mode='r',
                                  shape=self.real_shape)
            realization = np.array(real_mmap)
        else:
            realization = None

        fp = np.memmap(self.eta_file, dtype='float64', mode='w+',
                       shape=self.ndshape)
        surf = self.surface

        fp[0] = surf.surface_synthesis(realization, time=time, derivative=None)
        fp[1] = surf.surface_synthesis(realization, time=time, derivative='x')
        fp[2] = surf.surface_synthesis(realization, time=time, derivative='y')
        fp[3] = surf.surface_synthesis(realization, time=time, derivative='xx')
        fp[4] = surf.surface_synthesis(realization, time=time, derivative='xy')
        fp[5] = surf.surface_synthesis(realization, time=time, derivative='yy')

        fp.flush()


    def __del__(self):
        """delete temporary directory"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
