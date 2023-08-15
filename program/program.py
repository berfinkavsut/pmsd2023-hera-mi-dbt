from src.dataset import Dataset
from src.configuration import Configuration
from src.slab_generator import SlabGenerator

from typing import List, Dict, Any, Tuple

import os


class Program:
    """
    The program consists of three main objects: `Dataset`, `Configuration`, and `SlabGenerator`.
    These objects are designed to interact with each other to handle dataset
    information and manage the database for input and output operations.

    The child classes of Program class:
    - SlabGeneration
    - IQAMetrics
    - PerformanceMetrics

    """

    def __init__(self, settings_config: Dict[str, Any], dataset_config: Dict[str, Any] = None):

        # Setting configuration
        self._input_data_dir = settings_config['input-data-dir']
        self._output_data_dir = settings_config['output-data-dir']
        self._bit_depth = settings_config['bit-depth']
        self._iqa_metrics = settings_config['iqa-metrics']
        self._performance_metrics = settings_config['performance-metrics']

        # Dataset configuration
        self._dataset_config = dataset_config

        # Data frames
        self._df_series = None

        # Objects
        self._dataset = None
        self._configuration = None
        self._slab_generator = None

    def init(self):
        """
        Start the program and initialize the objects.
        """

        # Slab generation does not require `Dataset` class necessarily
        if self._dataset_config is not None:
            self._dataset = Dataset(config=self._dataset_config)
            self._df_series = self._dataset.get_series_df()
        else:
            self._dataset = None
            self._df_series = None

        self._configuration = Configuration(data_dir=self._output_data_dir)
        self._slab_generator = SlabGenerator()

    def filter_series(self, dataset: str, labels: List[str], series_instance_uids: List[str]):
        """
        Filter series by dataset type, labels, and SeriesInstanceUIDs.
        """
        return self._dataset.filter_series(dataset, labels, series_instance_uids)

    def filter_series_by_labels(self, labels: List[str], pos_labels: List[str], series_instance_uids: List[str]):
        """
        Filter series by labels and SeriesInstanceUIDs.
        Set labels to positive and negative.
        """
        return self._dataset.filter_series_by_labels(labels=labels,
                                                     pos_labels=pos_labels,
                                                     series_instance_uids=series_instance_uids)

    def read_config_json(self, config_hash_code: str):
        """
        Read JSON files by config hash code.
        Configurations are inside the output folder.
        """

        config_dir = os.path.join(self._output_data_dir, config_hash_code)
        config_json = f'config_{config_hash_code}.json'
        config_filepath = os.path.join(config_dir, config_json)
        return self._configuration.read_config_json(config_filepath=config_filepath)

    def list_dicom_filepaths(self):
        """
        List the DICOM files which are inside the input folder.
        """

        dir = self._input_data_dir
        filepaths = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.dcm'):
                    filepaths.append(os.path.join(root, file))
        filepaths.sort()
        return filepaths, len(filepaths)

    def list_config_hash_codes(self):
        """
        List the config hash codes which are inside the output folder.
        """

        dir = self._output_data_dir
        config_hash_codes = []
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                if dir not in ['iqa_metrics', 'performance_metrics', 'predictions']:
                    config_hash_codes.append(dir)
            break  # Stop iteration after the first level
        config_hash_codes.sort()
        return config_hash_codes

    def list_series_instance_uids(self, config_hash_code: str):
        """
        List the SeriesInstanceUIDs folders, which are inside
        the configuration folders having the name of config hash code.
        """

        config_dir = os.path.join(self._output_data_dir, config_hash_code)
        series_instance_uids = []
        for root, dirs, files in os.walk(config_dir):
            for dir in dirs:
                series_instance_uids.append(dir)
            break  # Stop iteration after the first level
        series_instance_uids.sort()
        return series_instance_uids
