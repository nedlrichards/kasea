import importlib

def load_surface(experiment):
    """import module with surface and initialize"""
    surface_type = experiment.toml_dict['surface']['type']
    module = importlib.import_module('kasea.surfaces.' + surface_type)
    # case insentive search for surface class
    all_vals = dir(module)
    # Module is file name before underscore
    class_name = (surface_type.split("_")[0]).casefold()
    class_i = [val.casefold() for val in all_vals].index(class_name)
    surf_class = getattr(module, all_vals[class_i])
    curr_instance = surf_class(experiment)
    return curr_instance
