import tomli
import numpy as np
import numexpr as ne


def parse_toml(toml_file):
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
