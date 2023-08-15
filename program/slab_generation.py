from program.program import Program
from src.dicom_file_processor import DICOMFileProcessor as dfp
from src.image_processor import ImageProcessor as ip

from typing import List, Dict, Any, Tuple

import numpy as np

import os
import cv2
import natsort


class SlabGeneration(Program):
    """
    SlabGeneration is used to read DICOM files, generate slab configurations, generate slabs, read/save slabs.

    """

    def read_dicom(self, dicom_filepath: str, current_number: int, total_number: int):
        """
        Read DICOM file by DicomFileProcessor class and return numpy object.
        """
        image, series_instance_uid = dfp.read_dicom(dicom_filepath=dicom_filepath,
                                                    current_number=current_number,
                                                    total_number=total_number)
        return image, series_instance_uid

    def generate_configurations(self, slab_configs: Dict[str, List[Any]], save_mode: bool = False):
        """
        Generate slab configurations by Configuration class.
        """
        return self._configuration.generate_configurations(slab_configs=slab_configs,
                                                           save_mode=save_mode)

    def generate_slabs(self, image: np.ndarray, config: Dict[str, Any]):
        """
        Generate slabs by SlabGenerator class and return numpy object.
        """
        return self._slab_generator.generate_slabs(image=image, config=config)

    def read_slabs(self, config_hash_code: str, series_instance_uid: str):
        """
        Read slabs which were already saved inside configuration folders
        with the name of SeriesInstanceUIDs.
        """

        config_dir = os.path.join('', *[self._output_data_dir, config_hash_code])
        slabs_dir = os.path.join('', *[config_dir, series_instance_uid])

        if os.path.exists(slabs_dir):
            slabs_dir_files = os.listdir(slabs_dir)
            slabs_set = []

            # Sort the slab filenames by ascending numbers
            for i, slab_filename in enumerate(natsort.natsorted(slabs_dir_files)):

                if not slab_filename.endswith('.png'):
                    continue

                slab_filepath = os.path.join(slabs_dir, slab_filename)
                slab_image = cv2.imread(slab_filepath, 0)

                # Rescale intensity inside the range of [0,1]
                slab_image = ip.min_max_normalization(slab_image)
                slabs_set.append(slab_image)

            if len(slabs_set) != 0:
                pixel_array = np.stack(slabs_set)
                return pixel_array

        return None

    def save_slabs(self, slabs: np.ndarray, config_hash_code: str, series_instance_uid: str):
        """
        Save slabs inside configuration folders with the name of SeriesInstanceUIDs.
        """

        config_output_dir = os.path.join('', *[self._output_data_dir, config_hash_code])
        slabs_output_dir = os.path.join('', *[config_output_dir, series_instance_uid])
        os.makedirs(slabs_output_dir, exist_ok=True)

        for i in range(slabs.shape[0]):
            slab_image = slabs[i, :, :]
            slab_image_path = os.path.join(slabs_output_dir, f"{series_instance_uid}_slab_{i}.png")

            # Save slab images with 16-bit depth using OpenCV
            # Note: 16-bit unsigned (CV_16U) images can be saved in the case of PNG, JPEG 2000, and TIFF formats
            slab_image = ip.rescale_intensity(image=slab_image, new_min_val=0.0, new_max_val=2 ** self._bit_depth - 1)
            slab_image = slab_image.astype(np.uint16)
            cv2.imwrite(filename=slab_image_path, img=slab_image)
