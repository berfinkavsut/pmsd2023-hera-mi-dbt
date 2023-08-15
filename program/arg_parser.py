import argparse
import yaml
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Slab Generation')

    parser.add_argument('--config_dataset',
                        default='config/dataset.yaml',
                        help='Path to the YAML file for dataset')

    parser.add_argument('--config_settings',
                        default='config/settings.yaml',
                        help='Path to the YAML file for settings')

    parser.add_argument('--config_slabs',
                        default='config/slab_configs.yaml',
                        help='Path to the YAML file for slab generation configurations')

    parser.add_argument('--root',
                        default=os.getcwd(),
                        help='Path to root')

    return parser.parse_args()


def load_settings(args):
    with open(args.config_settings, 'r') as file:
        settings_config = yaml.load(file, Loader=yaml.FullLoader)

    # Replace $ROOT$ placeholder by the provided --root value
    for cfg in settings_config:
        if isinstance(settings_config[cfg], str) and '$ROOT$' in settings_config[cfg]:
            settings_config[cfg] = settings_config[cfg].replace('$ROOT$', args.root)

    return settings_config


def load_dataset(args):
    with open(args.config_dataset, 'r') as file:
        dataset_config = yaml.load(file, Loader=yaml.FullLoader)

    # Replace $ROOT$ placeholder by the provided --root value
    for cfg in dataset_config:
        if isinstance(dataset_config[cfg], str) and '$ROOT$' in dataset_config[cfg]:
            dataset_config[cfg] = dataset_config[cfg].replace('$ROOT$', args.root)

    return dataset_config


def load_slab_configs(args):
    with open(args.config_slabs, 'r') as file:
        slab_generation_config = yaml.load(file, Loader=yaml.FullLoader)
    return slab_generation_config
