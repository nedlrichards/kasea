import importlib

def load_surface(experiment):
    """import module with surface and initialize"""
    surface_type = experiment.toml_dict['surface']['type']
    module = importlib.import_module('kasea.surfaces.' + surface_type)
    # case insentive search for surface class
    all_vals = dir(module)
    class_i = [val.casefold() for val in all_vals].index(surface_type.casefold())
    surf_class = getattr(module, all_vals[class_i])
    curr_instance = surf_class(experiment)
    return curr_instance
