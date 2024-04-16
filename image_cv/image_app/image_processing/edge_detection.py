import numpy as np
import cv2


def detect_edges(img: np.ndarray) -> np.ndarray:

    """
    Detects edges in an image using the Sobel filter.

    Applies separate Sobel filters to detect horizontal and vertical edges in the input image,
    and then combines these edges to produce a single edge-detected image. The Sobel operator
    calculates the gradient of the image intensity at each pixel, providing the directional change
    in the intensity or color.

    Parameters:
    img (numpy.ndarray): The input image for which edges are to be detected. Should be a grayscale image.

    Returns:
    numpy.ndarray: An image highlighting the edges detected from both horizontal and vertical directions.

    Note:
    This function expects the input image to be in grayscale format. For best results, convert any color
    image to grayscale before applying this function.
    """

    if img.size == 0:
        raise ValueError("Input image must not be empty")

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to get the derivatives in the x direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel filter to get the derivatives in the y direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to range 0 to 255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude


def blur_image(image: np.ndarray) -> np.ndarray:

    # Apply filter
    blurred_output = cv2.GaussianBlur(image, (7, 7), 0)
    return blurred_output

