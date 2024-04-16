from ..image_processing import detect_edges
import numpy as np
import cv2
from ..helper import create_test_grayscale_image, create_test_image

def test_detect_edges_on_uniform_image():

    """
    Verifies that the edge detection function correctly identifies that there are no edges
    in a uniform color image. This test ensures that the edge detection algorithm does not
    falsely detect edges in areas of uniform intensity.

    A 100x100 pixel image filled entirely with white (255) is used for the test. The expectation
    is that the edge detection function returns an image of the same size with no edge features
    detected, represented by all pixel values being zero.
    """

    # Create a uniform color image
    img = create_test_grayscale_image(100, 100, 255)  # All white image

    # Detect edges
    edges = detect_edges(img)

    # Since the image is uniform, no edges should be detected
    assert np.array_equal(edges, np.zeros_like(img)), "No edges should be detected in a uniform color image"


def test_detect_edges_on_synthetic_image():

    """
    Tests the edge detection on a synthetic 10x10 image with a vertical boundary,
    verifying that the detect_edges function identifies the boundary between a black and a white half.
    
    Asserts that detected edges exist along the expected vertical line, confirming the function's accuracy.
    """

    # Create a synthetic image with a vertical dividing line
    img = create_test_grayscale_image(10, 10, 0)
    img[:, 5:] = 255  # Half white, half black
    print(img)

    # Detect edges
    edges = detect_edges(img)

    # Check if edges are detected correctly along the boundary
    # Check a range of rows around the expected edge location to ensure the edge is detected
    detected_edge = np.any(edges[:, 4:5] > 0)  # Checking a small range around the expected line

    assert detected_edge, "Edges should be detected at the boundary"


def test_detect_edges_on_empty_image():
    """
    Tests the detect_edges function with an empty image to check if it raises a ValueError as expected.
    This helps ensure the function handles error conditions gracefully.
    """
    # Create an empty image
    img = np.array([]).reshape(0, 0)

    # Try to detect edges and check for the appropriate error
    try:
        edges = detect_edges(img)
        # If no error is raised, force the test to fail
        assert False, "detect_edges should raise an error for an empty image"
    except ValueError as ve:
        # Check if the raised exception is a ValueError
        assert isinstance(ve, ValueError), "detect_edges should raise a ValueError for an empty image"
    except Exception as e:
        # If any other exception is caught, fail the test and show the unexpected error type
        assert False, f"Unexpected exception type raised: {type(e).__name__}"


def test_detect_edges_on_striped_image():

    """
    Tests edge detection on an image with vertical stripes.

    This test verifies that the edge detection function identifies edges correctly in a synthetic
    image where vertical stripes alternate every 10 pixels between black and white. It checks for
    edge detection along the stripe boundaries across the image width.

    Asserts:
        The test ensures that edges are detected along each stripe boundary, confirming that the edge
        detection function effectively recognizes sudden changes in pixel intensity.
    """
    
    # Create a striped pattern image
    img = create_test_grayscale_image(100, 100, 0)
    img[:, ::10] = 255  # Vertical stripes every 10 pixels
    
    # Detect edges
    edges = detect_edges(img)
    
    # Check that edges are detected along the stripes
    for i in range(10, 100, 10):
        assert np.any(edges[:, i-1:i+1] > 0), f"Edges should be detected at stripe at column {i}"


def test_detect_edges_on_color_image():

    """
    Tests edge detection on a synthetic color image with a distinct boundary.

    This test ensures that the edge detection function can handle color images and correctly identifies
    edges where there is a sharp color contrast. A test image is created with a clear vertical boundary
    dividing two halves: one black and the other white.

    Asserts:
        The test checks that edges are prominently detected along the color boundary. It evaluates a small
        range around the expected boundary to ensure that the detected edges are located where the color changes.
    """
    
    # Create a color image with clear edges
    img = create_test_image(100, 100, (0, 0, 0))
    img[:, 50:, :] = [255, 255, 255]  # Half white, half black
    
    # Detect edges
    edges = detect_edges(img)
    
    # Check that edges are detected at the color boundary
    assert np.any(edges[:, 49:52] > 0), "Edges should be detected at the color boundary"