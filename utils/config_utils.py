import os
from dynaconf import Dynaconf


def load_config(config_file):
    settings = Dynaconf(settings_files=[config_file])
    return settings


def get_config(filename):
    config_file = f'{filename}.yaml'

    config = load_config(config_file)
    return config
