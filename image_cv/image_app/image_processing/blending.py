import numpy as np
import cv2


def blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:

    """
    Blends two images using Laplacian pyramids for multi-resolution blending.

    This function resizes the first image to match the dimensions of the second, 
    constructs Laplacian pyramids for both, and blends them based on a gradient mask 
    that transitions from the left to the right of the images.

    Parameters:
    img1 (numpy.ndarray): First image to blend.
    img2 (numpy.ndarray): Second image to blend with the first.

    Returns:
    numpy.ndarray: The resulting blended image.

    Notes:
    Both images should be in a compatible format (e.g., both 8-bit or floating point).
    """

    height2, width2 = img2.shape[:2]
    
    resized_img1 = cv2.resize(img1, (width2, height2), interpolation=cv2.INTER_AREA)
    resized_img2 = img2

    mask = np.zeros((height2, width2, 3), dtype=np.uint8)
    
    # Fill the right half of the mask with white (255)
    mask[:, width2 // 2:] = 255 
    mask = cv2.resize(mask, (width2, height2), interpolation=cv2.INTER_AREA)

    smaller_dim = min(resized_img1.shape[0], img1.shape[1])
    max_level = int(np.floor(np.log2(smaller_dim)))

    # Store the pyramid levels 
    gaussian_pyr = [resized_img1]
    gaussian_pyr2 = [resized_img2]
    mask_pyr = [mask]
    
    laplacian_pyr1 = []
    laplacian_pyr2 = []

    # Generate Gaussian and Laplacian pyramids for both images and the mask
    for _ in range(max_level):
        # Step 1=> Smoothen the image
        resized_img1 = cv2.GaussianBlur(resized_img1, (11, 11), 2)
        resized_img2 = cv2.GaussianBlur(resized_img2, (11, 11), 2)
        mask = cv2.GaussianBlur(mask, (11, 11), 2)

        # Downsample the images to half their sizes
        resized_img1 = cv2.resize(resized_img1, (resized_img1.shape[1] // 2, resized_img1.shape[0] // 2))
        resized_img2 = cv2.resize(resized_img2, (resized_img2.shape[1] // 2, resized_img2.shape[0] // 2))
        mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))

        # Store the pyramid levels
        gaussian_pyr.append(resized_img1)
        gaussian_pyr2.append(resized_img2)
        mask_pyr.append(mask)

    for i in range(len(gaussian_pyr) - 1): # At each pyramid level
        # Upsample the images
        upsampled1 = cv2.resize(gaussian_pyr[i + 1], (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        upsampled2 = cv2.resize(gaussian_pyr2[i + 1], (gaussian_pyr2[i].shape[1], gaussian_pyr2[i].shape[0]))

        # Get the laplacian by subtracting smoothen image from reconstructed original version of that level.
        sharpened1 = cv2.subtract(gaussian_pyr[i], upsampled1)
        sharpened2 = cv2.subtract(gaussian_pyr2[i], upsampled2)

        # Store the levels. 
        laplacian_pyr1.append(sharpened1)
        laplacian_pyr2.append(sharpened2)

    # At each level perform alpha blending 
    alpha_blended_pyramid = []

    for i in range(max_level): 
        alpha = mask_pyr[i] / 255.0  # Normalize values between [0,1]
        alpha_blended = (1 - alpha) * laplacian_pyr1[i] + alpha * laplacian_pyr2[i]
        alpha_blended_pyramid.append(alpha_blended)

    # Combine the blended images to reconstruct the original blended image. 
    blended_image = alpha_blended_pyramid[-1]

    for i in range(max_level - 1, -1, -1):
        upsampled = cv2.resize(blended_image, (alpha_blended_pyramid[i].shape[1], alpha_blended_pyramid[i].shape[0]))
        blended_image = alpha_blended_pyramid[i] + upsampled

    return blended_image
