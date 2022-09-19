import numpy as np
import numexpr as ne
from tempfile import mkdtemp
import shutil
from os import path

class Realization:
    """Class to create and manipulate surfaces on disk"""

    def __init__(self, surface, include_hessian=False, chunksize=1e5):
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

        self.x_a = surface.x_a
        self.y_a = surface.y_a

        # setup file to store surfaces
        if surface.y_a is None:
            if include_hessian:
                self.ndshape = (3, self.x_a.size)
            else:
                self.ndshape = (2, self.x_a.size)
        else:
            if include_hessian:
                self.ndshape = (6, self.x_a.size, self.y_a.size)
            else:
                self.ndshape = (3, self.x_a.size, self.y_a.size)

        self.include_hessian = include_hessian

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

        fp[0] = surf.surface_synthesis(realization, time=time, derivative=None)
        fp[1] = surf.surface_synthesis(realization, time=time, derivative='x')

        if surf.y_a is not None:
            fp[2] = surf.surface_synthesis(realization, time=time, derivative='y')
            save_ind = 3
        else:
            save_ind = 2

        if self.include_hessian:
            fp[save_ind] = surf.surface_synthesis(realization, time=time, derivative='xx')
            if self.y_a is not None:
                fp[save_ind + 1] = surf.surface_synthesis(realization, time=time, derivative='xy')
                fp[save_ind + 2] = surf.surface_synthesis(realization, time=time, derivative='yy')

        fp.flush()


    def integral_mask(self, theta, to_shadow=False):
        """Return surface positions within a total range of d_max"""
        fp = np.memmap(self.eta_file, dtype='float64', mode='r',
                       shape=self.ndshape)

        # setup delay bound calculation
        z_s = self.surface.z_src
        z_r = self.surface.z_rcr
        dr = self.surface.dr
        d_max = self.surface.tau_max * self.surface.c

        z = fp[0]
        x_r = r_rcr[0]
        z_r = r_rcr[-1]
        if self.y_a is None:
            x_a = self.x_a
            y_a = 0
            x_r = dr
            y_r = 0
        else:
            x_a = self.x_a[:, None]
            y_a = self.y_a[None, :]
            x_r = dr * np.cos(theta)
            y_r = dr * np.sin(theta)

        d_src = ne.evaluate("sqrt(x_a ** 2 + y_a ** 2 + (z - z_s) ** 2)")
        d_rcr = ne.evaluate("sqrt((x_r - x_a) ** 2 + (y_r - y_a) ** 2 + (z_r - z) ** 2)")
        distance_mask = (d <= d_max)
        if not to_shadow:
            return distance_mask

        # shadow test works on projected geometry
        proj_d_src = ne.evaluate("sqrt(x_a ** 2 + y_a ** 2")
        proj_d_rcr = ne.evaluate("sqrt((x_r - x_a) ** 2 + (y_r - y_a) ** 2)")

        # restrict showning domain
        eta = z[distance_mask]
        d_src = d_src[distance_mask]
        d_rcr = d_rcr[distance_mask]
        proj_d_src = proj_d_src[distance_mask]
        proj_d_rcr = proj_d_rcr[distance_mask]

        # sort bearings by projected source and receiver distance
        sort_i_src = np.argsort(proj_d_src, kind='heapsort')
        sort_i_rcr = np.argsort(proj_d_rcr, kind='heapsort')

        # define same launch angle sign for source and receiver
        th_src = (eta - z_s) / d_src
        th_rcr = (eta - z_r) / d_rcr

        if self.y_a is None:
            src_shad_i = self._shadow_1d(proj_d_src[sort_i_src], th_src[sort_i_src])
            rcr_shad_i = self._shadow_1d(proj_d_rcr[sort_i_rcr], th_rcr[sort_i_rcr])
            return (src_shad_i | rcr_shad_i)

        th_src = z_s / d_src
        th_rcr = z_r / d_rcr
        th_src = th_src[distance_mask]
        th_rcr = th_rcr[distance_mask]

        phi_src = ne.evaluate("(x_a * y_r - x_r * y_a) / (dr * sqrt(x_a ** 2 + y_a ** 2))")
        phi_rcr = ne.evaluate("((x_r - x_a) * y_r - x_r * (y_r - y_a)) / (dr * sqrt((x_a - x_r) ** 2 + (y_a - y_r) ** 2))")
        phi_src = phi_src[distance_mask]
        phi_rcr = phi_rcr[distance_mask]

        src_shad_i = self._shadow_2d(proj_d_src[sort_i_src], phi_src[sort_i_src], th_src[sort_i_src])
        rcr_shad_i = self._shadow_2d(proj_d_rcr[sort_i_rcr], phi_rcr[sort_i_rcr], th_rcr[sort_i_rcr])
        return (src_shad_i | rcr_shad_i)


    def _shadow_1d(self, proj_r, theta):
        """Create mask where either source or receiver ray is shadowed"""
        pass


    def _shadow_2d(self, proj_r, phi, theta):
        """Create mask where either source or receiver ray is shadowed"""
        pass


    def __del__(self):
        """delete temporary directory"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
