import json
import hashlib
import os

from typing import Any, Dict, List
from itertools import product


class Configuration:
    """
    Generate configurations by taking specific configuration parameters
    and producing their every possible combination.

    Save each dictionary of configuration in a JSON file.
    Every configuration is mapped by using unique hash codes.

    Slab configuration input keys are,
    - 'projection_method',
    - 'thickness_overlap',
    - 'slice_skip_ratio',
    - 'breast_skin_removal',
    - 'breast_skin_kernel_size',
    - 'breast_skin_iteration_number'

    Data folder structure with configurations is as follows:
    - data
        - input
            - series_instance_uid_1.dcm
            - series_instance_uid_2.dcm
            - ...
        - output
            - config_hash_code_1
                config_hash_code_1.json
                - series_instance_uid_1
                    series_instance_uid_1_slab_1.png
                    series_instance_uid_1_slab_2.png
                    ...
                - series_instance_uid_2
                - ...
            - config_hash_code_2
            - ...
    """

    def __init__(self, data_dir: str):

        # Output directory to save configurations and slab generation results
        self._data_dir = data_dir

    def generate_configurations(self, slab_configs: Dict[str, List[Any]], save_mode=False):
        """
        Generate the configurations from different parameters and
        save them in JSON files.

        :param slab_configs: keys of configurations, and possible configuration inputs
        :param save_mode: save configurations in JSON files or not
        :return: list of mapped hash codes and number of configurations
        """

        # Generate all possible combinations
        slab_config_keys = slab_configs.keys()

        # Generate all possible combinations
        # slab_configs_combinations = list(product(slab_configs['projection-method'],
        #                                         slab_configs['thickness-overlap'],
        #                                         slab_configs['breast-skin-removal'],
        #                                         slab_configs['breast-skin-kernel-size'],
        #                                         slab_configs['breast-skin-iteration-number'],
        #                                         ))

        slab_params = []
        for key in slab_config_keys:
            print(slab_configs[key])
            slab_params.append(slab_configs[key])

        # slab_params = [slab_configs[key] for key in slab_config_keys]
        slab_configs_combinations = list(product(*slab_params))

        slab_config_hash_codes = []
        for slab_config_elem in slab_configs_combinations:

            # zip() creates an iterable of key-value pairs
            # dict() creates a dictionary from the key-value pairs
            slab_config = dict(zip(slab_config_keys, slab_config_elem))
            slab_config_hash_code = self.get_config_hash_code(config=slab_config)

            if save_mode is True:
                self.save_config_json(config=slab_config)
            else:
                self.get_config_json_filepath(config=slab_config)

            slab_config_hash_codes.append(slab_config_hash_code)

        return slab_config_hash_codes, len(slab_configs_combinations)

    def save_config_json(self, config: Dict[str, Any]):
        """
        Save configuration in a JSON file inside its folder with the name of mapped hash code.

        :param config: dictionary of configuration
        :return: filepath of configuration
        """

        # Each configuration has one output folder
        config_output_dir, config_hash_code = self.make_output_dir(config=config)

        # Save JSON file with configuration parameters
        config_filepath = os.path.join(config_output_dir, f'config_{config_hash_code}.json')
        with open(config_filepath, 'w') as file:
            json.dump(config, file, indent=4)

        return config_filepath

    def make_output_dir(self, config: Dict[str, Any]):
        """
        Make the output directory, which has the name of mapped hash code of the configuration.
        Later, the configuration as JSON file and results form slab generation will be saved here.

        :param config: dictionary of configuration
        :return: output directory, mapped hash code for configuration
        """

        # Map the configuration to its hash code
        config_hash_code = self.get_config_hash_code(config=config)

        # Make output directory
        output_dir = os.path.join(self._data_dir, config_hash_code)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir, config_hash_code

    def get_config_hash_code(self, config: Dict[str, Any]):
        """
        Map the configuration to its hash code by using SHA256 algorithm.

        :param config: dictionary of configuration
        :return: mapped hash code
        """

        # Convert the dictionary to a JSON string
        json_data = json.dumps(config, sort_keys=True)

        # Generate the hash code using the SHA256 algorithm
        config_hash_code = hashlib.md5(json_data.encode()).hexdigest()

        return config_hash_code

    def read_config_json(self, config_filepath: str):
        """
        Read the JSON file and return configuration.

        :param config_filepath: filepath of JSON file containing configuration
        :return: dictionary of configuration
        """

        # Read the JSON file
        with open(config_filepath, 'r') as file:
            slab_config_dict = json.load(file)

        return slab_config_dict

    def get_config_json_filepath(self, config: Dict[str, Any]):
        """
        Get the JSON filepath of the configuration: output/{hash_code}/config_{hash_code}.JSON

        :param config: dictionary of configuration
        :return: filepath of JSON file
        """

        # Each configuration has one output folder
        config_output_dir, config_hash_code = self.make_output_dir(config=config)

        # Save JSON file with configuration parameters
        config_filepath = os.path.join(config_output_dir, f'config_{config_hash_code}.json')

        return config_filepath
