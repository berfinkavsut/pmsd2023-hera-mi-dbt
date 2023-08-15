import matplotlib.pyplot as plt
import numpy as np

from skimage import morphology
from skimage.filters import threshold_triangle, threshold_li
from skimage.transform import resize

from typing import Tuple


class ImageProcessor:
    """
    ImageProcessor has the image processing methods that are used for slab generation.
    Image processing methods are taken from the Python library "scikit-image".

    """
    def __init__(self):
        pass

    @staticmethod
    def histogram(image: np.ndarray, min_val, max_val, plot=False, plot_title="Image Histogram"):
        """
        Return the histogram of the image with bins, plot it if desired.

        :param image: 2D image
        :param min_val: minimum value for the histogram
        :param max_val: maximum value for the histogram
        :param plot: plot the histogram or not
        :param plot_title: title
        :return: histogram and bins
        """
        # Calculate the histogram
        histogram, bins = np.histogram(image.flatten(), bins='auto')

        if plot is True:
            plt.figure()
            plt.hist(image.flatten(), bins=(max_val - min_val + 1), range=[min_val, max_val], color='b', alpha=0.7)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(plot_title)
            plt.show()

        return [histogram, bins]

    @staticmethod
    def min_max_normalization(image: np.ndarray):
        """
        Rescale intensity inside the range of [0,1].

        :param image: 2D image
        :return: normalized 2D image
        """
        if (np.max(image) - np.min(image)) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return image

    @staticmethod
    def rescale_intensity(image: np.ndarray, new_min_val: float, new_max_val: float):
        """
        Rescale intensity inside the range of [minimum value, maximum value].

        :param image: 2D image with the range of [0,1]
        :param new_min_val: minimum value of the new range
        :param new_max_val: maximum value of the new range
        :return: rescaled 2D image
        """
        image = image * (new_max_val - new_min_val) + new_min_val
        return image

    @staticmethod
    def rescale(image: np.ndarray, scale_factor: float = 1):
        """
        Resize the image with the scaling factor.

        :param image: 2D image
        :param scale_factor: scaling factor in x- and y-axis
        :return: resized 2D image
        """
        sh = image.shape
        row = int(sh[0] * scale_factor)
        col = int(sh[1] * scale_factor)

        image_rescaled = ImageProcessor.resize(image=image, output_shape=(row, col))
        return image_rescaled

    @staticmethod
    def resize(image: np.ndarray, output_shape: Tuple[int, int] = None):
        """
        Resize the image with the output shape.

        :param image: 2D image
        :param output_shape: output size
        :return: resized 2D image
        """
        if output_shape is None:
            output_shape = image.shape

        image_resized = resize(image=image,
                               output_shape=output_shape,
                               anti_aliasing=True,
                               preserve_range=True)
        return image_resized

    @staticmethod
    def binary_threshold(image: np.ndarray, threshold_val, inverse=False):
        """
        Apply binary thresholding.

        :param image: 2D image
        :param threshold_val: threshold value for comparison
        :param inverse: inverse of the result image
        :return:
        """

        # Apply binary thresholding
        if inverse:
            binary = image < threshold_val
        else:
            binary = image > threshold_val

        # Convert the binary image to a mask with 0 and 1 values
        return binary.astype(np.float64)

    @staticmethod
    def triangle_threshold(image: np.ndarray, inverse=False):
        """
        Apply triangle thresholding.

        :param image: 2D image
        :param inverse: inverse of the result image
        :return: thresholded image mask
        """

        # Apply triangle thresholding
        threshold_val = threshold_triangle(image)

        if inverse:
            binary = image < threshold_val
        else:
            binary = image > threshold_val

        # Convert the binary image to a mask with 0 and 1 values
        return binary

    @staticmethod
    def li_threshold(image: np.ndarray, inverse=False):
        """
        Apply Li thresholding.

        :param image: 2D image
        :param inverse: inverse of the result image
        :return: thresholded image mask
        """

        # Apply Li thresholding
        threshold_val = threshold_li(image)

        if inverse:
            binary = image < threshold_val
        else:
            binary = image > threshold_val

        # Convert the binary image with True, False values
        return binary

    @staticmethod
    def erode(image: np.ndarray, kernel_size: int, kernel_structure: str = 'disk', iter_num: int = 1):
        """
        Erode the 2D image with a kernel having a desired size.
        If iter_num is not 1, then repeat this operation as much as iter_num times.

        :param image: 2D image
        :param kernel_size: kernel size of the morphological structure
        :param kernel_structure: disk or square
        :param iter_num: iteration number
        :return: eroded image
        """

        # Select kernel
        kernel = morphology.disk(radius=kernel_size)

        if kernel_structure == 'square':
            kernel = morphology.square(width=kernel_size)

        # Perform erosion on the image
        for i in range(iter_num):
            image = morphology.erosion(image, kernel)

        return image

    @staticmethod
    def dilate(image: np.ndarray, kernel_size: int, iter_num: int = 1):
        """
        Dilate the 2D image with a rectangular kernel with desired kernel size.
        If iter_num is not 1, then repeat this operation as much as iter_num times.

        :param image: 2D image
        :param kernel_size: kernel size of the morphological structure
        :param iter_num: iteration number
        :return: dilated image
        """

        # Perform dilation on the image
        kernel = morphology.disk(radius=kernel_size)

        for i in range(iter_num):
            image = morphology.dilation(image, kernel)

        return image

    @staticmethod
    def closing(image: np.ndarray, kernel_size: int):
        """
        Closing is a morphological operation performing first dilation then erosion.
        Closes small gaps and holes.

        :param image: 2D image
        :param kernel_size: kernel size of the morphological structure
        :return: closed image
        """

        kernel = morphology.disk(radius=kernel_size)
        image_closing = morphology.closing(image, kernel)

        return image_closing

    @staticmethod
    def opening(image: np.ndarray, kernel_size: int):
        """
        Opening is a morphological operation performing first erosion then dilation.
        Removes small noise and smoothens the object boundaries.

        :param image: 2D image
        :param kernel_size: kernel size of the morphological structure
        :return: opened image
        """

        kernel = morphology.disk(radius=kernel_size)
        image_opening = morphology.opening(image, kernel)

        return image_opening

    @staticmethod
    def soft_thresholding(image: np.ndarray, threshold_val: float):
        """
        Image has non-negative pixel values.
        f(x) = x - threshold_val if x > threshold_val,
               0 if x < threshold_val

        :param image: 2D image
        :param threshold_val: threshold value to clip the pixel intensity values
        :return: soft thresholded 2D image
        """
        return np.maximum(np.abs(image) - threshold_val, 0)

    @staticmethod
    def pad_image(image, pad_width: int):
        """
        Pad the image with the desired width by 'symmetric' mode.

        :param image: 2D image
        :param pad_width: width to pad around the image (in pixels)
        :return: padded image
        """

        # Calculate the pad widths for each dimension
        pad_widths = [(pad_width, pad_width)] * image.ndim

        # Pad the image symmetrically
        padded_image = np.pad(image, pad_widths, mode='symmetric')

        return padded_image
