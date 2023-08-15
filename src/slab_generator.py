from src.image_processor import ImageProcessor
from src.projection import Projection
from src.test import *

from typing import Any, Dict
from skimage.exposure import match_histograms
from skimage import exposure

import numpy as np
import math


class SlabGenerator:
    """
    SlabGenerator takes 3D digital breast tomosynthesis image and
    generates 2D slab images with these projection methods:
    - maximum intensity projection,
    - average intensity projection,
    - soft MIP.

    When breast skin removal mode is on, the breast skin is segmented out from the projected slab images.
    Kernel size for the breast skin segmentation is configurable.

    Notes:
    - Slabs have thickness and overlap parameters in slice numbers.
    It is assumed that slices have 1 mm distance between each other.

    - Last slice images are discarded if they are not inside
    the slice indices of the last generated slab.
    """
    def __init__(self):

        # 3D DBT image
        self.image = None

        # Projection
        self._projection = Projection()

        # Image Processing
        self._image_processor = ImageProcessor()

        # Default configuration parameters
        self._config = {
            "projection_method": "mip",
            "thickness_overlap": [0, 10],
            "slice_skip_ratio": [0.0, 0.0],
            "breast_skin_removal": 0,
            "breast_skin_kernel_size": 5,
            "breast_skin_iteration_number": 1,
        }

        self._projection_method = None
        self._thickness = None  # in slice numbers
        self._overlap = None  # in slice numbers
        self._slice_skip_ratio = None
        self._breast_skin_removal = None
        self._breast_skin_kernel_size = None
        self._breast_skin_iter_num = None

        # For projections with breast skin removal
        self._breast_image_pixel_num = None

    def generate_slabs(self, image: np.ndarray, config: Dict[str, Any]):
        """
        Configure slab generation, preprocess slice images, generate slab indices,
        and generate slab images with the slab indices with projection method.
        Clear configured parameters after slabs are generated.

        :return: stacked slab images as 3D image
        """

        self.image = image
        # check_image_range(image)

        # 1. Configure
        self._configure(config)

        # 2. Preprocessing: background suppression
        self._preprocess()

        # 3. Generate slab indices
        slab_indices, slab_num = self._generate_slabs_indices()

        # 4. Generate slabs
        slabs = np.zeros(shape=(slab_num, self.image.shape[1], self.image.shape[2]))

        for slab_id, indices in slab_indices.items():
            start_idx, end_idx = indices
            slab_slices = self.image[start_idx:end_idx, :, :]

            # 5. Apply projection for each slab
            slab_image = self._project_slabs(slab_slices, start_idx, end_idx)
            # check_image_range(slab_image)

            # 6. Add slab to the list of slabs
            slabs[slab_id, :, :] = slab_image

        self._clear_parameters()

        return slabs

    def _clear_parameters(self):
        """
        Clear the parameters after generating the slabs.
        So that, SlabGenerator object can be used several times with different configurations.

        :return: None
        """

        self.image = None

        self._projection_method = None
        self._thickness = None
        self._overlap = None
        self._slice_skip_ratio = None
        self._breast_skin_removal = None
        self._breast_skin_kernel_size = None
        self._breast_skin_iter_num = None

        self._breast_image_pixel_num = None

    def _configure(self, config: Dict[str, Any]):
        """
        Update the configuration parameters with the input configuration.
        Take the parameters of configuration to set individually.

        :param config: dictionary of configuration parameters
        :return: None
        """

        if config is not None:
            self._config.update(config)

        self._projection_method = self._config["projection_method"]
        self._thickness, self._overlap = self._config["thickness_overlap"]
        self._slice_skip_ratio = self._config["slice_skip_ratio"]
        self._breast_skin_removal = self._config["breast_skin_removal"]
        self._breast_skin_kernel_size = self._config["breast_skin_kernel_size"]
        self._breast_skin_iter_num = self._config["breast_skin_iteration_number"]

    def _preprocess(self):
        """
        In the preprocessing step, we clear background noise for each slice by using the background mask,
        which is computed by applying triangle thresholding.

        If breast skin removal mode is on, save the number of non-zero pixels in each slice image.
        This information will be used to choose the largest slice image during projecting the slabs.

        :return: preprocessed 3D image
        """

        if self._breast_skin_removal == 2:
            self._breast_image_pixel_num = np.zeros(shape=(self.image.shape[0], 1))

        # Background suppression is implemented by default
        for i in range(self.image.shape[0]):

            self.image[i, :, :] = self._remove_background_noise(image=self.image[i, :, :])
            # check_image_range(self.image[i, :, :])

            if self._breast_skin_removal == 2:
                _, _, background_mask = self._breast_segmentation(image=self.image[i, :, :],
                                                                  kernel_size=self._breast_skin_kernel_size,
                                                                  iter_num=self._breast_skin_iter_num)
                breast_mask = (1.0-background_mask)
                self._breast_image_pixel_num[i] = np.sum(breast_mask)

        return self.image

    def _generate_slabs_indices(self):
        """
        Generate slab indices by using the parameters "thickness" and "overlap".
        Save the start and end slice indices in a dictionary with the keys of slab indices.

        Example 1: thickness = 6, overlap = 3
        slab_0: i_0 = 0,            j_0 = i_0 + 6 = 6,   [0, 6] -> slice indices: [0, 1, 2, 3, 4, 5]
        slab_1: i_1= j_0 - 3 = 2,   j_1 = i_1 + 6 = 8,   [2, 8] -> slice indices: [2, 3, 4, 5, 6, 7]
        ...
        slab_indices = {0: [0, 5], 1: [2, 7], ...}

        Example 2: thickness = 5, overlap = 0. slice_skip_ratio = [0.25, 0.25], slab_num = 100
        skip_start = 100 * 0.25 = 25 -> 25 slices are discarded from the top!
        skip_end = 100 - 100 * 0.25 = 75  ->  25 slices are discarded from the bottom!

        slab_0: i_0 = 25,          j_0 = i_0 + 5 = 30,   [25, 30] -> slice indices: [25, 26, 27, 28, 29]
        slab_1: i_1= j_0 - 0 = 30, j_1 = i_1 + 5 = 35,   [30, 35] -> slice indices: [30, 31, 32, 33, 34]
        ...
        slab_9: i_9= j_9 - 0 = 70, j_9 = i_9 + 5 = 75,   [70, 75] -> slice indices: [70, 71, 72, 73, 74]
        slab_indices = {0: [25, 30], 1: [30, 35], ...}

        :return: slab_indices
        """

        slab_num = self.image.shape[0]
        skip_start = math.floor(self._slice_skip_ratio[0] * slab_num)
        skip_end = slab_num - math.floor(self._slice_skip_ratio[1] * slab_num)

        slab_indices = {}
        count = 0
        i = skip_start
        while (i + self._thickness) <= skip_end:
            j = i + self._thickness  # not included
            slab_indices[count] = (i, j)

            i = j - self._overlap
            count += 1

        slab_num = count
        return slab_indices, slab_num

    def _project_slabs(self, image: np.ndarray, start_idx: int, end_idx: int):
        """
        Project the slice images between start and end indices.
        Projection methods are mip, aip, and soft mip.

        If the breast skin removal mode is 0, then there is no post-processing for breast skin.
        If the breast skin removal mode is 1, segment out the breast skin from the projected slab image.
        If the breast skin removal mode is 2, get the largest slice image between the start and end slice indices.
        Then, replace the breast skin of the projected slab with the breast skin of the slice image.

        :param image: 3D image with 2D slice images
        :return: 2D projected slab image
        """

        projection_image = self._projection.project(image=image, method=self._projection_method)

        if self._breast_skin_removal == 0:
            return projection_image

        elif self._breast_skin_removal == 1:

            # Breast without skin (from projection image)
            _, breast_skin_mask, _ = self._breast_segmentation(image=projection_image,
                                                               kernel_size=self._breast_skin_kernel_size,
                                                               iter_num=self._breast_skin_iter_num)
            projection_image, _ = self._remove_breast_skin(image=projection_image, breast_skin_mask=breast_skin_mask)
            return projection_image

        elif self._breast_skin_removal == 2:

            # Breast without skin (from projection image)
            _, breast_skin_mask, _ = self._breast_segmentation(image=projection_image,
                                                               kernel_size=self._breast_skin_kernel_size,
                                                               iter_num=self._breast_skin_iter_num)

            breast_image_orig, _ = self._remove_breast_skin(image=projection_image, breast_skin_mask=breast_skin_mask)

            # Breast skin (from the largest slice image)
            breast_image_pixel_num = self._breast_image_pixel_num[start_idx:end_idx]
            largest_slice_image_idx = start_idx + np.argmax(breast_image_pixel_num)
            largest_slice_image = self.image[largest_slice_image_idx, :, :]

            _, breast_skin_image = self._remove_breast_skin(image=largest_slice_image, breast_skin_mask=breast_skin_mask)

            # Equalize the histograms of both the breast skin and breast images
            breast_skin_image = self._image_processor.min_max_normalization(exposure.equalize_hist(breast_skin_image))
            breast_image = self._image_processor.min_max_normalization(exposure.equalize_hist(breast_image_orig))

            # Match histograms of the new image and original image
            image = breast_image + breast_skin_image
            image = match_histograms(image=image, reference=projection_image)

            # Take only the breast skin after histogram matching
            breast_skin_image_new = image * breast_skin_mask
            projection_image = breast_image_orig + breast_skin_image_new

        return projection_image

    def _remove_background_noise(self, image: np.ndarray):
        """
        Segment the breast mask of the image and set the background pixels to zero.

        :param image: 2D slice image
        :return: image with zero background
        """

        # Breast segmentation (with breast skin)
        background_suppress_mask = self._image_processor.triangle_threshold(image=image)
        background_suppress_mask = background_suppress_mask.astype(np.float64)
        # check_mask(background_suppress_mask)

        # Background suppression
        image = image * background_suppress_mask
        return image

    def _remove_breast_skin(self, image: np.ndarray, breast_skin_mask: np.ndarray):
        """
        Remove breast skin by using the breast skin mask.

        :param image: 2D slice image
        :param breast_skin_mask: mask image having 0/1 values
        :return: image after breast skin removal and the removed breast skin image
        """

        breast_skin_image = breast_skin_mask * image
        image = (1.0-breast_skin_mask) * image
        return image, breast_skin_image

    def _breast_segmentation(self, image: np.ndarray, kernel_size: int = 10, iter_num: int = 1):
        """
        Segment breast, breast skin, and background masks having 0/1 values.
        Triangle thresholding is used to segment the breast part of the image.
        Then, the morphological operation "erosion" is used to segment the breast skin.
        Erosion is applied after lowering the resolution of the input image to reduce computational time.

        :param image: 2D image
        :param kernel_size: kernel size used for breast skin segmentation
        :param iter_num: iteration number used for breast skin segmentation
        :return: breast mask, breast skin mask, background mask
        """

        # 1. Breast segmentation (with breast skin)
        breast_with_skin_mask = self._image_processor.triangle_threshold(image=image)
        breast_with_skin_mask = breast_with_skin_mask.astype(np.float64)
        # check_mask(breast_with_skin_mask)

        # 2. Breast segmentation (without breast skin)
        breast_mask_down = self._image_processor.rescale(breast_with_skin_mask, scale_factor=0.25)

        # Pad the image in case the breast mask is at the edge of the slice image
        breast_mask_padded, view_lat = self._pad_image(image=breast_mask_down, pad_width=kernel_size)

        # Segment the skin by applying the morphological operation 'erosion'
        breast_mask = self._image_processor.erode(image=breast_mask_padded,
                                                  kernel_size=kernel_size,
                                                  iter_num=iter_num)

        # Discard the padded side after erosion
        if view_lat == 'R':
            breast_mask = breast_mask[kernel_size:-1].copy()
        elif view_lat == 'L':
            breast_mask[0:-kernel_size] = breast_mask[0:-kernel_size].copy()

        breast_mask = self._image_processor.resize(breast_mask, output_shape=image.shape)

        # Set all values larger than 0 to 1, necessary due to the interpolation of resizing
        breast_mask[breast_mask > 0.0] = 1.0
        # check_mask(breast_mask)

        # 3. Breast skin segmentation
        breast_skin_mask = np.abs(breast_with_skin_mask - breast_mask)

        # 4. Background segmentation
        background_mask = 1.0 - breast_with_skin_mask

        # check_mask(background_mask)
        # check_mask(breast_mask)
        # check_mask(breast_skin_mask)

        return [breast_mask, breast_skin_mask, background_mask]

    def _pad_image(self, image: np.ndarray, pad_width):
        """
        Find the view laterality of the 2D slice image
        and pad the image as much as the desired pad width on the opposite side.

        :param image: 2D image
        :return: "R" or "L" representing the right or left view laterality
        """

        image_padded = np.zeros(shape=(image.shape[0], image.shape[1] + pad_width))

        left_edge = np.sum(image[:, 0])    # sum of left edge pixels
        right_edge = np.sum(image[:, -1])  # sum of right edge pixels
        view_laterality = 'r' if left_edge < right_edge else 'l'

        if view_laterality == 'r':
            image_padded[:, pad_width:] = image
        elif view_laterality == 'l':
            image_padded[:, :-pad_width] = image

        return image_padded, view_laterality