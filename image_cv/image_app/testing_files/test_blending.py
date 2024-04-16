from ..image_processing import blend
import numpy as np
import cv2
from ..helper import create_test_image

def test_blend_same_size_images():

    """
    Tests the blend function to ensure it correctly blends two images of the same size.
    The test checks that the output image has the same dimensions as the input images.
    """

    # Create two images of the same size
    img1 = create_test_image(100, 100, (255, 0, 0))  # Red image
    img2 = create_test_image(100, 100, (0, 0, 255))  # Blue image

    # Blend the images
    result = blend(img1, img2)

    # Check if the result is of the correct shape
    assert result.shape == img1.shape, "Blended image must have the same shape as input images."


def test_blend_with_empty_image():

    """
    Tests the blend function with an empty image to check if it raises a ValueError as expected.
    """

    img1 = create_test_image(100, 100, (255, 255, 255))  # White image
    img2 = np.array([]).reshape(0, 0, 3)  # Empty image

    # Check behavior when one image is empty
    try:
        result = blend(img1, img2)
    except Exception as e:
        assert isinstance(e, ValueError), "Should raise a ValueError for empty input images"\


def test_blend_different_size_images():
    """
    Tests the blend function with two images of different sizes to verify that the output image
    matches the dimensions of the second (smaller) image as per the resizing logic in the blend function.
    """

    # Create two test images of different sizes using solid colors
    img1 = create_test_image(200, 200, (255, 0, 0))  # Red image of size 200x200
    img2 = create_test_image(100, 100, (0, 255, 0))  # Green image of size 100x100

    # Call the blend function
    result = blend(img1, img2)

    # Check that the output size matches the size of the second image
    # Using pytest assertion style
    assert result.shape == img2.shape, "Output image shape should match the size of the second input image"

