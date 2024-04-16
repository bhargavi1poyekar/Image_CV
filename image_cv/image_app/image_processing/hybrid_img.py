import numpy as np
import cv2


def create_hybrid_img(img1:np.ndarray, img2:np.ndarray) -> np.ndarray:

    """
    Creates a hybrid image by combining the low frequency components of one image
    with the high frequency components of another. The function resizes both images to the same dimensions,
    applies a Gaussian blur to extract low frequency components from the first image,
    and subtracts a Gaussian blur from the second image to extract high frequency components.

    Parameters:
    img1 (numpy.ndarray): The first image, from which low frequency components are extracted.
    img2 (numpy.ndarray): The second image, from which high frequency components are extracted.

    Returns:
    numpy.ndarray: The resulting hybrid image which combines visual elements from both images.

    Note:
    Kernel size and sigma values for Gaussian blurs are crucial and might need adjustment based on specific use cases.
    """
    
    height2, width2 = img2.shape[:2]
    
    resized_img1 = cv2.resize(img1, (width2, height2), interpolation=cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, (width2, height2), interpolation=cv2.INTER_AREA)

    low_freq_image = resized_img1
    high_freq_image = resized_img2

    # Apply Low pass filter to cat 
    lf_blurred = cv2.GaussianBlur(low_freq_image, (11, 11), 8)

    # (11,11)=> Kernel Size, I tried various sizes to see the different hybrids. With lower kernel sizes,
    # The smoothening was not that good. With higher kernel size, I got more smoothening effect.

    # 8 => stands for standard deviation (sigma). # greater the sigma=> Averages pixel values over wider area. 

    # Apply high pass filter to panda 
    hf_blurred = cv2.GaussianBlur(high_freq_image, (11, 11), 2)

    # We get high pass filter by subtracting the low pass filtered version from original image
    hf_filtered = high_freq_image - hf_blurred

    # Combine the images
    hybrid_image = lf_blurred + hf_filtered

    return hybrid_image