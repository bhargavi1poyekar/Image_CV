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

    # Sobel filter for gradients in x-direction
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    # Sobel filter for gradients in y_direction
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # Sobel filter for both horizontal and vertical edges
    edge_vertical = cv2.filter2D(img, -1, sobel_x)
    # Parameters=> (image, depth, kernel) => depth: -1 output image depth same as input image.
    edge_horizontal = cv2.filter2D(img, -1, sobel_y)
    
    # Combine the results to detect edges in all directions
    edge_output = cv2.addWeighted(edge_vertical, 0.5, edge_horizontal, 0.5, 0)
    
    return edge_output


def blur_image(image: np.ndarray) -> np.ndarray:

    # Apply filter
    blurred_output = cv2.GaussianBlur(image, (7, 7), 0)
    return blurred_output

