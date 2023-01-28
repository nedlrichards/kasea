import tomllib
import numpy as np
import numexpr as ne
from math import pi
import sys
import importlib.util

class XMission:
    """Load up a test scenario"""
    def __init__(self, toml_file):
        """scatter calculation specification load and basic setup"""
        self.toml_file = toml_file
        with open(toml_file, "rb") as f:
            toml_dict = tomllib.load(f)

        self.z_src = toml_dict['geometry']['zsrc']
        self.z_rcr = toml_dict['geometry']['zrcr']
        self.dr = toml_dict['geometry']['dr']

        tf = self._tf_specification(toml_dict)

        self.c, self.fc, self.fs, self.max_dur = tf[:4]
        self.max_surf_dur, self.t_a_pulse, self.f_a_pulse, self.pulse = tf[4:]

        self.t_a, self.f_a = self._tf_axes(toml_dict)
        self.dt = (self.t_a[-1] - self.t_a[0]) / (self.t_a.size - 1)

        self.pulse_FT = np.fft.rfft(self.pulse, self.t_a_pulse.size)

        # use image arrival to determine end of time axis
        r_img = np.sqrt(self.dr ** 2 + (self.z_src + self.z_rcr) ** 2)

        self.tau_img = r_img / self.c
        self.tau_max = self.tau_img + self.max_dur

        if 'theta' in toml_dict['surface']:
            self.theta = np.deg2rad(toml_dict['surface']['theta'])
        else:
            self.theta = 0.

        self.time_step = toml_dict['surface']['time_step'] if 'time_step' in toml_dict['surface'] else None
        self.num_steps = toml_dict['surface']['num_steps'] if 'num_steps' in toml_dict['surface'] else 1
        self.seed = toml_dict['surface']['seed'] if 'seed' in toml_dict else 0
        self.solutions = toml_dict['geometry']['solutions'] if 'solutions' in toml_dict['geometry'] else ['all']
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
        numf = t_a_pulse.size // 2 + 1
        dt = (t_a_pulse[-1] - t_a_pulse[0]) / (t_a_pulse.size - 1)
        fs = 1 / dt
        f_a_pulse = np.arange(numf) * fs / t_a_pulse.size

        return c, fc, fs, max_dur, max_surf_dur, t_a_pulse, f_a_pulse, pulse

    def _tf_axes(self, toml_dict):
        """Define the time and frequency axes"""
        if 't_pad' in toml_dict['t_f']:
            num_front_pad = int(np.ceil(toml_dict['t_f']['t_pad'] * self.fs))
        else:
            num_front_pad = 1

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

        return taxis, faxis
