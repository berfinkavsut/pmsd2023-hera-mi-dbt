from src.image_processor import ImageProcessor as ip

import os
import cv2

import numpy as np
import pydicom as dicom


class DICOMFileProcessor:
    """
    DICOMFileProcessor is used to read and extract DICOM files.

    Pixel Coordinates:
    3D images are defined in pixel coordinates with DICOM files.
    3D images' pixel arrays have their pixel values with the order of (z,y,x)-axes.

            z-axis
            ^
           /
          /
         --------> x-axis
         |
         |
         |
         y-axis

    """

    def __init__(self):
        pass

    @staticmethod
    def read_dicom(dicom_filepath: str, current_number: int = 1, total_number: int = 1):
        """
        Read the DICOM file, preprocess its pixel array, which is a 3D image.
        Return its pixel array and SeriesInstanceUID to track the DICOM file.

        :param dicom_filepath: DICOM filepath to read the file
        :return: 3D image from the DICOM and SeriesInstanceUID
        """

        if not os.path.exists(dicom_filepath):
            return None

        print("Reading {}... ({}/{})".format(dicom_filepath, current_number, total_number))

        # UserWarning: The(0028, 0101) Bits stored value '10' in the dataset does not match
        # the component precision value '16' found in the JPEG 2000 data_processing
        # It is recommended that you change the bits stored value to produce the correct output
        dicom_imageset = dicom.dcmread(dicom_filepath)

        try:
            dicom_imageset.decompress(handler_name="pylibjpeg")
        except:
            pass

        # Get the SeriesInstanceUID
        series_instance_uid = str(dicom_imageset.SeriesInstanceUID)

        # Normalize the pixel array to the range of [0,1]
        pixel_array = dicom_imageset.pixel_array
        pixel_array = ip.min_max_normalization(pixel_array).astype(np.float64)

        return pixel_array, series_instance_uid

    @staticmethod
    def extract_dicom(dicom_filepath: str, destination_dir: str, current_number: int = 1, total_number: int = 1):
        """
        Read the DICOM file and save their slice images as 16-bit gray scaled PNG files.
        This method is not necessary, only self.read_dicom() is used in our project.

        :param dicom_filepath: DICOM filepath to read the file
        :param destination_dir: destination directory to save the slice images from the DICOM files
        :return: None
        """

        # Read the DICOM file
        pixel_array, series_instance_uid = DICOMFileProcessor.read_dicom(dicom_filepath)
        slice_image_num = pixel_array.shape[0]

        # Create "SeriesInstanceUID" directory under the DICOM folder
        image_dir = os.path.join(destination_dir, series_instance_uid)
        os.makedirs(image_dir, exist_ok=True)

        print("Extracting {}... ({}/{})".format(dicom_filepath, current_number, total_number))

        for i in range(slice_image_num):
            slice_image = pixel_array[i, :, :]
            slice_image = ip.rescale_intensity(image=slice_image,
                                               new_min_val=0,
                                               new_max_val=2**16-1)
            slice_image = slice_image.astype(np.uint16)

            slice_image_path = os.path.join(image_dir, f"{i}.png")

            if os.path.exists(slice_image_path):
                continue

            cv2.imwrite(filename=slice_image_path, img=slice_image)

    @staticmethod
    def is_image_laterality_correct(pixel_array: np.ndarray, view_laterality: str):
        """
        Flip images if the real view laterality and recorded view laterality are not the same.
        View laterality can be "rcc", "lcc", "rmlo", and "lmlo".
        This method is not used in our project.

        :param pixel_array: 3D image
        :param view_laterality: 'r' or 'l' for the right or left view laterality
        :return: whether the image laterality was correct or not
        """

        left_edge = np.sum(pixel_array[:, :, 0])  # sum of left edge pixels
        right_edge = np.sum(pixel_array[:, :, -1])  # sum of right edge pixels
        image_laterality = "r" if left_edge < right_edge else "l"

        if image_laterality != view_laterality:
            return False

        return True