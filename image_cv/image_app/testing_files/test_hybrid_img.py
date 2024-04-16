from ..image_processing import create_hybrid_img
import numpy as np
import cv2
from ..helper import create_test_grayscale_image, create_test_image

def test_create_hybrid_image():
    """
    Tests the hybrid image creation by blending a low-frequency component from a uniform image
    with a high-frequency component from an image with a sharp edge. Verifies that the resulting
    hybrid image retains the low-frequency background and the sharp edge details.
    """
    # Create a blue image for low frequency details
    low_freq_img = create_test_image(100, 100, (255, 0, 0))  # Red image

    # Create a grayscale image with a vertical white stripe for high frequency details
    high_freq_img = create_test_grayscale_image(100, 100, 0)  # Black image
    high_freq_img[:, 45:55] = 255  # White stripe

    # Convert high_freq_img to BGR if needed by the hybrid function
    high_freq_img = cv2.cvtColor(high_freq_img, cv2.COLOR_GRAY2BGR)

    # Create the hybrid image
    hybrid_img = create_hybrid_img(low_freq_img, high_freq_img)

    # Check that the output has the expected color profile and stripe
    # Expected: stripe should be visible clearly, and the rest of the image should be red
    assert np.all(hybrid_img[:, 45:55] > 100), "Stripe should be visible in the hybrid image"
    assert np.all(hybrid_img[0, 0] == [255, 0, 0]), "The rest of the image should maintain the low-frequency red color"
