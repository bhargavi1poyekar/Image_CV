from django.core.files.storage import FileSystemStorage
from django.core.exceptions import ValidationError
from .image_processing import swap_phases, create_hybrid_img, blend, detect_keypoint, detect_edges, blur_image
import os
import cv2
from django.core.files import File
from typing import Tuple
import numpy as np

def validate_image(file:File)-> None:

    """
    Validates an image file based on its extension and size.
    
    Parameters:
    file (File): The file object to validate.

    Raises:
    ValidationError: If the file extension isn't supported or the file size exceeds 10 MB.
    """

    if not file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValidationError("Unsupported file extension.")
    if file.size > 10485760:  # 10MB
        raise ValidationError("File too large. Size should not exceed 10 MB.")
    

def delete_expired_files(path: str) -> None:

    """
    Deletes a file at a specified path if it exists.
    
    Parameters:
    path (str): The file path to check and potentially delete.
    """

    if os.path.isfile(path):
        os.remove(path)


def save_img(image_file: File) -> Tuple[str, str]:

    """
    Saves an image file to the filesystem and returns its URL and file path.
    
    Parameters:
    image_file (File): The image file to save.

    Returns:
    tuple: A tuple containing the URL of the uploaded file and its file path.

    Raises:
    Exception: If the file could not be saved.
    """
     
    try:
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)
        filepath = fs.path(filename)
        return uploaded_file_url, filepath
    except Exception as e:
        raise Exception(f"Failed to save original image")


def pre_process_img(image_file: File) -> Tuple[str, str, np.ndarray]:

    """
    Validates and saves an image file, then reads it back into memory.
    
    Parameters:
    image_file (File): The image file to process.

    Returns:
    tuple: A tuple containing the URL of the uploaded file, its file path, and the image array.

    Raises:
    ValidationError: If the image file fails validation.
    Exception: If the image file fails to pre-process.
    """

    try:
        validate_image(image_file)
        uploaded_file_url, filepath = save_img(image_file)
        orig_img = cv2.imread(filepath)
    except ValidationError as ve:
        raise ValidationError("Unsupported file extension.")
    except Exception as e:
        raise Exception(f"Failed to Pre-Process Image")

    return uploaded_file_url, filepath, orig_img


def post_process_img(filepath: str, processed_image: np.ndarray) -> str:

    """
    Processes and saves an image, returning the URL of the processed image.
    
    Parameters:
    filepath (str): The original file path of the image.
    processed_image (numpy.ndarray): The image data to save.

    Returns:
    str: URL of the processed image.

    Raises:
    Exception: If the image fails to process or save.
    """

    try:
        processed_filename = change_filename(filepath)
        processed_file_url = cv2_write(processed_filename, processed_image)
    except Exception as e:
        raise Exception(f"Failed to Pre-Process Image")
    return processed_file_url


def change_filename(filename: str) -> str:

    """
    Modifies a filename to denote that the image has been processed.
    
    Parameters:
    filename (str): The original filename of the image.

    Returns:
    str: A new filename indicating the image has been processed.
    """

    if '.jpg' in filename:
        processed_filename = filename.replace('.jpg', '_processed.jpg')
    elif '.png' in filename:
        processed_filename = filename.replace('.png', '_processed.png')
    elif '.jpeg' in filename:
        processed_filename = filename.replace('.jpeg', '_processed.jpeg')
    else:
        processed_filename = ''
    return processed_filename

    
def cv2_write(processed_filename: str, processed_image: np.ndarray) -> str:
    
    """
    Writes an image to the filesystem using OpenCV and returns its URL.
    
    Parameters:
    processed_filename (str): The filename under which to save the image.
    processed_image (numpy.ndarray): The image data to save.

    Returns:
    str: URL of the saved image.

    Raises:
    Exception: If the image cannot be processed or saved.
    """

    try:
        fs = FileSystemStorage()
        cv2.imwrite(processed_filename, processed_image)
        processed_file_url = fs.url(processed_filename.replace(fs.base_location, ''))
        return processed_file_url
    except Exception as e:
        raise Exception(f"Failed to process and save image")  


def create_test_image(width: int, height: int, color: tuple) -> np.ndarray:

    """
    Utility function to create a solid color image of specified dimensions and color.
    
    Parameters:
    width (int): The width of the image.
    height (int): The height of the image.
    color (tuple): A tuple representing the BGR color (Blue, Green, Red) of the image.
    
    Returns:
    np.ndarray: A NumPy array representing the image.
    """
    return np.full((height, width, 3), color, dtype=np.uint8)

def create_test_grayscale_image(width: int, height: int, intensity: int) -> np.ndarray:
    """
    Utility function to create a solid grayscale image of specified dimensions and intensity.
    
    Parameters:
    width (int): The width of the image.
    height (int): The height of the image.
    intensity (int): The grayscale intensity value for the image, from 0 (black) to 255 (white).
    
    Returns:
    np.ndarray: A NumPy array representing the grayscale image, with one channel.
    """
    return np.full((height, width), intensity, dtype=np.uint8)