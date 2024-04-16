from unittest import mock
import numpy as np
import cv2
from ..helper import validate_image, delete_expired_files, save_img, post_process_img, pre_process_img, change_filename, cv2_write
from django.core.exceptions import ValidationError
import pytest


def test_validate_image_valid_extension_and_size():
    """
    Tests that the `validate_image` function does not raise an exception for an image file with both a valid
    extension and an acceptable size. This ensures that the function correctly handles valid input without errors.
    """

    # Simulate a file with a valid extension and size
    mock_file = mock.Mock()
    mock_file.name = "test_image.jpg"
    mock_file.size = 1048570  # 1 MB

    # No exception should be raised for a valid file
    try:
        validate_image(mock_file)
    except ValidationError:
        pytest.fail("ValidationError raised unexpectedly!")


def test_validate_image_invalid_extension():

    """
    Tests that the `validate_image` function raises a ValidationError for an image file with an invalid extension.
    This ensures that the function correctly enforces file extension constraints.
    """

    # Simulate a file with an invalid extension
    mock_file = mock.Mock()
    mock_file.name = "test_image.bmp"
    mock_file.size = 1048570  # 1 MB

    # Expecting a ValidationError due to invalid extension
    with pytest.raises(ValidationError) as excinfo:
        validate_image(mock_file)
    assert "Unsupported file extension" in str(excinfo.value)


def test_validate_image_invalid_size():
    """
    Tests that the `validate_image` function raises a ValidationError for an image file that exceeds the maximum
    allowed file size. This checks the function's ability to enforce size limits.
    """

    # Simulate a file with a valid extension but invalid size
    mock_file = mock.Mock()
    mock_file.name = "large_test_image.jpg"
    mock_file.size = 20971520  # 20 MB

    # Expecting a ValidationError due to file size
    with pytest.raises(ValidationError) as excinfo:
        validate_image(mock_file)
    assert "Size should not exceed 10 MB" in str(excinfo.value)


def test_validate_image_invalid_size_and_extension():
    """
    Tests that the `validate_image` function raises a ValidationError for an image file with both an invalid
    extension and an excessive size. This test ensures the function correctly identifies and prioritizes the first
    encountered error (in this case, file extension) over subsequent ones.
    """

    # Simulate a file with both invalid size and extension
    mock_file = mock.Mock()
    mock_file.name = "large_test_image.bmp"
    mock_file.size = 20971520  # 20 MB

    # Only the first validation error (extension) should be raised
    with pytest.raises(ValidationError) as excinfo:
        validate_image(mock_file)
    assert "Unsupported file extension" in str(excinfo.value)

def test_change_filename_with_jpg():
    """
    Tests changing a .jpg filename to indicate the image has been processed.
    """
    filename = 'example.jpg'
    expected_filename = 'example_processed.jpg'
    assert change_filename(filename) == expected_filename, "Filename should be modified correctly for .jpg files"

def test_change_filename_with_png():
    """
    Tests changing a .png filename to indicate the image has been processed.
    """
    filename = 'example.png'
    expected_filename = 'example_processed.png'
    assert change_filename(filename) == expected_filename, "Filename should be modified correctly for .png files"

def test_change_filename_with_jpeg():
    """
    Tests changing a .jpeg filename to indicate the image has been processed.
    """
    filename = 'example.jpeg'
    expected_filename = 'example_processed.jpeg'
    assert change_filename(filename) == expected_filename, "Filename should be modified correctly for .jpeg files"

def test_change_filename_with_unsupported_extension():
    """
    Tests the filename modification function with an unsupported file extension,
    expecting no change to the filename.
    """
    filename = 'example.txt'
    expected_filename = ''
    assert change_filename(filename) == expected_filename, "Filename should not be modified for unsupported extensions"