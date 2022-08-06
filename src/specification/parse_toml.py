import tomli
import numpy as np
import numexpr as ne


def parse_toml(toml_file):
    """load a surface scatter specification file"""
    with open(toml_file, "rb") as f:
        toml_dict = tomli.load(f)
    return toml_dict
