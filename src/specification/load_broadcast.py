import tomli
import numpy as np
import numexpr as ne
from math import pi
import sys
import importlib.util

from src.surfaces import Surface

class Broadcast:
    """Load up a test scenario"""
    def __init__(self, toml_file):
        """scatter calculation specification load and basic setup"""
        self.toml_file = toml_file
        toml_dict = load_broadcast(toml_file)
        self.est_z_max = toml_dict['surface']['z_max']

        self.src, self.rcr = self._positions(toml_dict)
        tf = self._tf_specification(toml_dict)

        self.c, self.fc, self.fs, self.max_dur = tf[:4]
        self.max_surf_dur, self.t_a_pulse, self.pulse = tf[4:]

        tf = self._tf_axes(toml_dict)
        self.t_a, self.f_a = tf[:2]
        self.surf_t_a, self.surf_f_a = tf[2:]

        self.pulse_FT = np.fft.rfft(self.pulse, self.surf_t_a.size)

        # use image arrival to determine end of time axis
        r_img_2 = np.sum((self.src[:-1] - self.rcr[:-1]) ** 2) \
                  + (self.src[-1] + self.rcr[-1]) ** 2

        self.tau_img = np.sqrt(r_img_2) / self.c
        self.tau_max = self.tau_img + self.max_dur

        # axes and surface specification
        self.dx = self.c / (self.fs * toml_dict['surface']['decimation'])
        self.toml_dict = toml_dict


    def _positions(self, toml_dict):
        """Source and receiver postions"""
        src = np.zeros(toml_dict['geometry']['num_dim'] + 1)
        rcr = np.zeros(toml_dict['geometry']['num_dim'] + 1)
        src[-1] = toml_dict['geometry']['zsrc']
        rcr[-1] = toml_dict['geometry']['zrcr']

        if 'xsrc' in toml_dict['geometry']: src[0] = toml_dict['geometry']['xsrc']
        if 'xrcr' in toml_dict['geometry']: rcr[0] = toml_dict['geometry']['xrcr']

        if toml_dict['geometry']['num_dim'] == 2:

            if 'ysrc' in toml_dict['geometry']: src[1] = toml_dict['geometry']['ysrc']
            if 'yrcr' in toml_dict['geometry']: rcr[1] = toml_dict['geometry']['yrcr']

        return src, rcr


    def _tf_specification(self, toml_dict):
        """Setup of position axes and transmitted pulse"""
        c = toml_dict['t_f']['c']
        fc = toml_dict['t_f']['fc']
        fs = toml_dict['t_f']['fs']
        max_dur = toml_dict['t_f']['max_dur']

        if 'max_surf_dur' in toml_dict['t_f']:
            max_surf_dur = toml_dict['t_f']['max_surf_dur']
        else:
            max_surf_dur = max_dur

        xmitt = toml_dict['t_f']['pulse']
        xmitt += '_pulse'

        spec = importlib.util.spec_from_file_location("pulse",
                                             "experiments/" + xmitt + ".py")
        module = importlib.util.module_from_spec(spec)
        sys.modules['pulse'] = module
        spec.loader.exec_module(module)

        t_a_pulse, pulse = module.pulse(fc, fs)

        return c, fc, fs, max_dur, max_surf_dur, t_a_pulse, pulse

    def _tf_axes(self, toml_dict):
        """Define the time and frequency axes"""
        if 't_pad' in toml_dict['t_f']:
            num_front_pad = int(np.ceil(toml_dict['t_f']['t_pad'] * self.fs))
        else:
            num_front_pad = int(np.ceil(0.1e-3 * self.fs))

        num_back_pad = self.t_a_pulse.size

        # specifications of receiver time series
        numt = int(np.ceil(self.fs * self.max_dur)) \
             + num_front_pad + num_back_pad
        if numt % 2: numt += 1

        # compute time and frequency axes
        dt = 1 / self.fs
        taxis = np.arange(numt) * dt
        taxis -= dt * num_front_pad

        numf = numt // 2 + 1
        faxis = np.arange(numf) * self.fs / numt

        # specifications of surface time series
        numt = int(np.ceil(self.fs * self.max_surf_dur)) \
             + num_front_pad + num_back_pad
        if numt % 2: numt += 1

        # compute time and frequency axes
        surf_taxis = np.arange(numt) * dt
        surf_taxis -= dt * num_front_pad

        numf = numt // 2 + 1
        surf_faxis = np.arange(numf) * self.fs / numt

        return taxis, faxis, surf_taxis, surf_faxis

    def _setup_surface(self, toml_dict):
        """Use scatter duration to bound surface integration"""

        return surf, seed


def load_broadcast(toml_file):
    """load a surface scatter specification file"""
    with open(toml_file, "rb") as f:
        toml_dict = tomli.load(f)

    # parse time frequency specifications
    items = toml_dict['t_f'].items()
    t_f = {}
    for k, v in items:

        if k in ['pulse']:
            # these keys are not parsed by numexpr
            t_f[k] = v
        else:
            t_f[k] = ne.evaluate(v)[()]

    toml_dict['t_f'] = t_f

    # parse soource receiver geometry specifications
    items = toml_dict['geometry'].items()
    toml_dict['geometry'] = {k: ne.evaluate(v)[()] for k, v in items}

    # parse scatter surface specifications
    items = toml_dict['surface'].items()
    surface = {}
    for k, v in items:

        if k in ['type']:
            # these keys are not parsed by numexpr
            surface[k] = v
        elif k == 'theta':
            try:
                surface[k] = ne.evaluate(v)[()]
            except TypeError:
                num_list = v.strip('[]').split(',')
                num_list = np.array([ne.evaluate(n.strip()) for n in num_list])
                surface[k] = num_list
        else:
            surface[k] = ne.evaluate(v)[()]

    toml_dict['surface'] = surface

    return toml_dict
