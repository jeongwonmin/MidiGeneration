import yaml


def load_settings(setting_path):
    params = {}
    with open(setting_path, 'r') as f:
        params = yaml.load(f)
    return params
