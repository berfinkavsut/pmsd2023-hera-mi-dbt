from src.image_processor import ImageProcessor as ip
from src.test import *

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


class IQA:
    """
    IQA is used to compute the image quality assessment metrics for the generated slabs.
    IQA metrics are SNR, CNR, and contrast.

    When IQA metrics are computed for 3D images with slice images,
    the average and standard deviation of the metrics are returned.
    """

    def __init__(self):
        pass

    @staticmethod
    def _extract_roi_images(image: np.ndarray, object_coord: Dict[str, Any], plot_mode: bool = False):
        """
        Extract ROI images and background images.
        For background image, Li thresholding is used in the segmented breast image to represent the healthy tissue.

        :param image: 2D image
        :param object_coord: object coordinates to compute the CNR/contrast inside
        :return: object images, background image
        """

        # Object mask
        box_num = len(object_coord['x'])
        object_mask = np.zeros(shape=(box_num, *image.shape), dtype=np.float64)

        # Create separate masks for each box
        for i in range(box_num):
            x = object_coord['x'][i]
            y = object_coord['y'][i]
            object_mask[i, y[0]:y[1], x[0]:x[1]] = 1

        # Object image
        object_imgs = image * object_mask

        # Background mask: thresholded image * inverse image of the sum of the object masks
        breast_mask = ip.triangle_threshold(image=image)
        breast_mask_th = np.zeros_like(breast_mask)
        breast_mask_th[breast_mask] = ip.li_threshold(image=image[breast_mask], inverse=True)
        background_mask = breast_mask_th * (1 - np.sum(object_mask, axis=0))

        # Background image
        background_img = image * background_mask

        if plot_mode:
            IQA._plot_objects(object_imgs)
            IQA._plot_masks(breast_mask, breast_mask_th, background_mask, background_img)

        return object_imgs, background_img

    @staticmethod
    def contrast_to_noise_ratio(image: np.ndarray, object_coord: Dict[str, Any], plot_mode: bool = False):
        """
              ( Mean pixel value of object - Mean pixel value of the background )
        CNR = ------------------------------------------------------------------
               Square root of the sum of variances of the object and background

        Object is the diseased tissue and background is the healthy tissue.
        Source: Thick Slices from Tomosynthesis Data Sets: Phantom Study for the Evaluation of Different Algorithms
        Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3043718/#CR17

        Note: For micro-calcifications, CNR is not used due to having too few pixels.
        Instead of CNR, contrast is preferred to assess.

        :param image: 2D image
        :param object_coord: object coordinates to compute the CNR inside
        :return: CNR values from object coordinates
        """

        # Extract ROIs
        object_imgs, background_img = IQA._extract_roi_images(image, object_coord, plot_mode)

        # Mean values
        mean_background = np.mean(background_img[background_img > 0])

        box_num = len(object_coord['x'])
        mean_object = np.zeros(shape=(box_num, 1))
        for i in range(box_num):
            x = object_coord['x'][i]; y = object_coord['y'][i]
            object_img = image[y[0]:y[1], x[0]:x[1]]
            mean_object[i] = np.mean(object_img)

        # Standard deviation values
        std_background = np.std(background_img[background_img > 0])

        std_total = np.zeros(shape=(box_num, 1))
        for i in range(box_num):
            x = object_coord['x'][i]; y = object_coord['y'][i]
            object_img = image[y[0]:y[1], x[0]:x[1]]
            std_object = np.std(object_img)

            std_total[i] = np.sqrt(std_background**2 + std_object**2)

        # Compute CNR
        # Set CNR to infinity for zero division case
        cnr = np.zeros(shape=(box_num, 1))
        for i in range(box_num):
            if std_total[i] == 0:
                cnr[i] = np.inf
            else:
                cnr[i] = np.divide((mean_object[i] - mean_background), std_total[i])

        return cnr

    @staticmethod
    def contrast(image: np.ndarray, object_coord: Dict[str, Any], plot_mode: bool = True):
        """
              ( Mean pixel value of object - Mean pixel value of the background )
        C = ---------------------------------------------------------------------
              ( Mean pixel value of object + Mean pixel value of the background )

        Object is the diseased tissue and background is the healthy tissue.
        Source: Thick Slices from Tomosynthesis Data Sets: Phantom Study for the Evaluation of Different Algorithms
        Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3043718/#CR17

        :param image: 2D image
        :param object_coord: object mask to compute the contrast inside
        :return: contrast value
        """

        # Extract ROIs
        object_imgs, background_img = IQA._extract_roi_images(image, object_coord, plot_mode)

        # Mean values
        box_num = len(object_coord['x'])
        mean_object = np.zeros(shape=(box_num, 1))
        for i in range(box_num):
            x = object_coord['x'][i];
            y = object_coord['y'][i]
            object_img = image[y[0]:y[1], x[0]:x[1]]
            mean_object[i] = np.mean(object_img)

        mean_background = np.mean(background_img[background_img > 0])

        # Compute contrast
        contrast = np.divide((mean_object - mean_background), (mean_object + mean_background))

        return contrast

    @staticmethod
    def contrast_to_noise_ratio_3d(image: np.ndarray, object_coords: List[Dict[str, Any]], plot_mode: bool = False):
        """
        Compute the CNR for all the slab images of the 3D image.

        :param image: 3D image with slab images
        :param object_coords: object mask to compute the CNR inside
        :return: CNR values, mean and std of CNR values from the slab images
        """

        cnr_values = []
        for object_coord in object_coords:
            slab_idx = object_coord['slab']
            cnr = IQA.contrast_to_noise_ratio(image[slab_idx, :, :], object_coord, plot_mode=plot_mode)
            cnr_values.append({'slab': slab_idx,
                               'cnr': cnr.tolist(),
                               'x': object_coord['x'],
                               'y': object_coord['y']}
                              )

        cnr_list = np.concatenate([cnr_slab['cnr'] for cnr_slab in cnr_values])
        mean_cnr = np.mean(cnr_list).tolist()
        std_cnr = np.std(cnr_list).tolist()

        return cnr_values, mean_cnr, std_cnr

    @staticmethod
    def contrast_3d(image: np.ndarray, object_coords: List[Dict[str, Any]], plot_mode: bool = False):
        """
        Compute the contrast for all the slab images of the 3D image.

        :param image: 3D image with slab images
        :param object_coords:  object mask to compute the CNR inside
        :return: contrast values, mean and std of contrast values from the slab images
        """

        contrast_values = []
        for object_coord in object_coords:
            slab_idx = object_coord['slab']
            contrast = IQA.contrast(image[slab_idx, :, :], object_coord, plot_mode=plot_mode)
            contrast_values.append({'slab': slab_idx,
                                    'contrast': contrast.tolist(),
                                    'x': object_coord['x'],
                                    'y': object_coord['y']}
                                   )

        contrast_list = np.concatenate([contrast_slab['contrast'] for contrast_slab in contrast_values])
        mean_contrast = np.mean(contrast_list).tolist()
        std_contrast = np.std(contrast_list).tolist()

        return contrast_values, mean_contrast, std_contrast

    @staticmethod
    def _plot_masks(breast_mask, breast_mask_th, background_mask, background_img):

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(breast_mask.astype(np.float64), cmap='gray', vmin=0, vmax=1)
        plt.title('Breast Mask')

        plt.subplot(2, 2, 2)
        plt.imshow(breast_mask_th, cmap='gray', vmin=0, vmax=1)
        plt.title('Breast Mask after Li Thresholding')

        plt.subplot(2, 2, 3)
        plt.imshow(background_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Background Mask')

        plt.subplot(2, 2, 4)
        plt.imshow(background_img, cmap='gray', vmin=0, vmax=1)
        plt.title('Background Image')
        plt.show()

    @staticmethod
    def _plot_objects(object_img):
        object_num = object_img.shape[0]

        plt.figure(figsize=(10, 8))
        for i in range(object_num):
            plt.subplot(1, object_num, i+1)
            plt.title(f'Object image #{i+1}')
            plt.imshow(object_img[i, :, :], cmap='gray', vmin=0, vmax=1)
        plt.show()
