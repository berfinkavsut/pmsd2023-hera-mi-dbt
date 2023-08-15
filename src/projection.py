import numpy as np


class Projection:
    """
    Projection has the projection methods which are used for slab generation.
    Projection methods are maximum intensity projection, average intensity projection, and soft MIP.
    """

    def __init__(self):
        """
        Available projection methods are;
        - "mip": Maximum Intensity Projection
        - "aip": Average Intensity Projection
        - "soft_mip": Soft Maximum Intensity Projection

        New projection methods can be added to `projection function map`.
        """

        #####################################################################################
        self.projection_function_map = {
            "mip": self.maximum_intensity_projection,
            "aip": self.average_intensity_projection,
            "soft_mip": self.soft_mip,
            # Add more projection methods here as needed.
        }
        #####################################################################################

    def project(self, image: np.ndarray, method: str):
        """
        Generate slabs from a part of a 3D DBT image using the selected projection method.

        :param image: part of the 3D DBT image to be projected
        :param method: The selected projection method. It can be one of the following:

        :return: projected image, i.e. 2D slab image
        """

        projection_image = np.zeros(shape=(image.shape[1], image.shape[2]))

        if method in self.projection_function_map:
            projection_function = self.projection_function_map[method]
            projection_image = projection_function(image)

        return projection_image

    @staticmethod
    def maximum_intensity_projection(image: np.ndarray):
        """
        Calculate the maximum intensity projection along the z-axis.
        Take the maximum pixel value along each projection line.

        :param image: 3D image
        :return: maximum intensity projected image, 2D image
        """

        mip_image = np.max(image, axis=0)
        return mip_image

    @staticmethod
    def average_intensity_projection(image: np.ndarray):
        """
        Calculate the average intensity projection along the z-axis
        Average the pixel values along each projection line.

        :param image: 3D image
        :return: average intensity projected image, 2D image
        """

        aip_image = np.mean(image, axis=0)
        return aip_image

    @staticmethod
    def soft_mip(image: np.ndarray):
        """
        Calculate the soft maximum intensity projection along the z-axis
        Weighting function of the softMIP, f(x) = x⁴ where 0 <= x <= 1, is taken from this article:

        Source: "softMip: a novel projection algorithm for ultra-low-dose computed tomography
        Link: https://pubmed.ncbi.nlm.nih.gov/17955296/

        :param image: 3D image
        :return: soft mip image, 2D image
        """

        # Weighting function with range of [0, 1]
        n = image.shape[0]
        x = np.linspace(0, 1, n, endpoint=True)
        weighting_func = np.power(x, 4)

        # Rescale the projection to keep the maximum value constant
        scale = np.ones(shape=(image.shape[1], image.shape[2]))
        weighting_func_sum = np.sum(weighting_func)
        scale = np.multiply(scale, np.divide(1, weighting_func_sum))

        # Reshape the weighting function for broadcasting
        weighting_func_reshaped = weighting_func.reshape(-1, 1, 1)

        # Create projection lines
        sorted_image = np.sort(image, axis=0)

        # Compute the SoftMIP projection
        weighted_image = weighting_func_reshaped * sorted_image
        soft_mip_image = scale * np.sum(weighted_image, axis=0)

        return soft_mip_image
