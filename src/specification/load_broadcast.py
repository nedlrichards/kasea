import tomli
import numpy as np
import numexpr as ne
from math import pi
import sys
import importlib.util

from src.specification.parse_toml import parse_toml


class Broadcast:
    """Load up a test scenario"""
    def __init__(self, toml_file):
        """scatter calculation specification load and basic setup"""
        self.toml_file = toml_file
        toml_dict = parse_toml(toml_file)

        self.z_src = toml_dict['geometry']['zsrc']
        self.z_rcr = toml_dict['geometry']['zrcr']
        self.dr = toml_dict['geometry']['dr']

        tf = self._tf_specification(toml_dict)

        self.c, self.fc, self.fs, self.max_dur = tf[:4]
        self.max_surf_dur, self.t_a_pulse, self.pulse = tf[4:]

        tf = self._tf_axes(toml_dict)
        self.t_a, self.f_a = tf[:2]
        self.surf_t_a, self.surf_f_a = tf[2:]

        self.pulse_FT = np.fft.rfft(self.pulse, self.surf_t_a.size)

        # use image arrival to determine end of time axis
        r_img = np.sqrt(self.dr ** 2 + (self.z_src + self.z_rcr) ** 2)

        self.tau_img = r_img / self.c
        self.tau_max = self.tau_img + self.max_dur

        # specular point is center of theta rotation
        self.x_img = self.z_src * self.dr / (self.z_src + self.z_rcr)
        self.theta = toml_dict['surface']['theta'] if 'theta' in toml_dict['surface'] else 0.

        # axes and surface specification
        self.dx = self.c / (self.fs * toml_dict['surface']['decimation'])
        self.toml_dict = toml_dict


    def _tf_specification(self, toml_dict):
        """Setup of position axes and transmitted pulse"""
        c = toml_dict['t_f']['c']
        fc = toml_dict['t_f']['fc']
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

        t_a_pulse, pulse = module.pulse(fc)
        dt = (t_a_pulse[-1] - t_a_pulse[0]) / (t_a_pulse.size - 1)
        fs = 1 / dt

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
