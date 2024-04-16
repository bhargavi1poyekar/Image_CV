import numpy as np
import cv2
from typing import List, Tuple


def swap_phases(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Swaps the phase information of the Fourier transforms between two images and reconstructs the images
    from these modified Fourier transforms. This technique is often used in signal processing to explore the
    importance of phase and magnitude in image data.

    Parameters:
    img1 (numpy.ndarray): The first input image.
    img2 (numpy.ndarray): The second input image. Both images should be the same size or will be resized to match.

    Returns:
    tuple: A tuple of two numpy.ndarrays representing the images reconstructed after the phase swap. The first
           element corresponds to the first image with the phase of the second, and vice versa.

    Note:
    The function uses logarithmic scaling and normalization to enhance the visibility of Fourier magnitude 
    spectra before phase swapping. Images are returned in their original spatial dimensions, and pixel values 
    are scaled to 8-bit unsigned integers.
    """

    height2, width2 = img2.shape[:2]
    
    resized_img1 = cv2.resize(img1, (width2, height2), interpolation=cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, (width2, height2), interpolation=cv2.INTER_AREA)


    fft_img1 = np.fft.fft2(resized_img1, axes=(0, 1))
    fft_img2 = np.fft.fft2(resized_img2, axes=(0, 1))

    # Find magnitude and phase of the ffts
    mag1, phase1 = np.abs(fft_img1), np.angle(fft_img1)
    mag2, phase2 = np.abs(fft_img2), np.angle(fft_img2)

    # Apply logarithmic scaling to magnitude values
    # I had to do this, because magnitude values were very big and I was not able to display them
    log_mag1 = np.log(1 + mag1)
    log_mag2 = np.log(1 + mag2)

    # Normalize the scaled magnitude to get the values between [0, 1]
    scaled_log_mag1 = (log_mag1 - np.min(log_mag1)) / (np.max(log_mag1) - np.min(log_mag1))
    scaled_log_mag2 = (log_mag2 - np.min(log_mag2)) / (np.max(log_mag2) - np.min(log_mag2))

    # Swapping phase of both imags
    swapped_img1 = np.abs(mag1) * np.exp(1j * phase2)
    swapped_img2 = np.abs(mag2) * np.exp(1j * phase1)

    # Find ifft to regenerate the images
    regenerated_img1 = np.fft.ifft2(swapped_img1, axes=(0, 1)).real.astype(np.uint8)
    regenerated_img2 = np.fft.ifft2(swapped_img2, axes=(0, 1)).real.astype(np.uint8)

    return regenerated_img1, regenerated_img2