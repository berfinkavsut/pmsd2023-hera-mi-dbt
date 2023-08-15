import numpy as np
from src.projection import Projection
from src.image_processor import ImageProcessor


def check_image_range(image: np.ndarray):
    assert np.min(image) >= 0, "Image values should be greater than or equal to 0."
    assert np.max(image) <= 1, "Image values should be less than or equal to 1."


def check_projection_range(method: str):
    """
    Examples:
    check_projection(method='aip')
    check_projection(method='mip')
    check_projection(method='soft_mip')
    """

    image = np.ones(shape=(10, 1000, 1000))

    ip = ImageProcessor()
    image_normalized = ip.min_max_normalization(image)

    proj = Projection()
    image_projection = proj.project(image=image_normalized, method=method)

    assert np.abs(np.max(image_projection) - 1) < 1e-3, "Projection should preserve the range of pixel values."


def check_mask(mask: np.ndarray):
    assert set(mask.flatten()) == {0, 1}, "Mask should contain only 0 and 1 values."


def check_std(image, std):
    std_real = np.abs(np.std(image.flatten()))
    assert np.abs(std_real - std) < 1e-3
