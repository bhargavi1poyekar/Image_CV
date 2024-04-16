import cv2
import numpy as np
from typing import List, Tuple


def compute_h_matrix(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Computes the components of the Hessian matrix for an image, using Sobel operators to calculate
    the gradients and applying Gaussian weights for smoothing. This function returns the second
    moment matrix elements which are useful for feature detection tasks like corner detection.

    Parameters:
    image (numpy.ndarray): The input image for which the Hessian matrix components are to be computed.
                           The image should ideally be in grayscale format.

    Returns:
    tuple: A tuple containing three numpy.ndarrays, representing the weighted sums of the squares 
           and product of the image derivatives:
           (Ix2, Iy2, Ixy)
           where Ix2 is the Gaussian-weighted sum of the square of the x-gradient,
           Iy2 is the Gaussian-weighted sum of the square of the y-gradient,
           and Ixy is the Gaussian-weighted sum of the product of the x and y gradients.

    Note:
    This function uses a Sobel operator of size 3x3 and a Gaussian kernel of 5x5 with a sigma of 0.5.
    These parameters can affect the sensitivity of the result to image features.
    """

    # Using 3x3 Sobel operator to compute the x,y derivatives. 
    # CV_64F => datatype of float64
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Applying Circularly Symmetric Weights => Gaussian of 5x5, sigma=0.5
    gauss_matrix = cv2.getGaussianKernel(5, 0.5)

    W = np.outer(gauss_matrix, gauss_matrix.T)

    # Wp*Ix^2 => H[0,0]
    Ix2 = cv2.filter2D(Ix * Ix, -1, W)

    # Wp*Iy^2 => H[1,1]
    Iy2 = cv2.filter2D(Iy * Iy, -1, W)

    # Wp*Ix^Iy => H[0,1], H[1,0]
    Ixy = cv2.filter2D(Ix * Iy, -1, W)
    
    return Ix2, Iy2, Ixy


# Task 2: Corner Strength Function
def corner_strength(Ix2: np.ndarray, Iy2: np.ndarray, Ixy: np.ndarray) -> np.ndarray:

    """
    Calculates the corner strength of an image based on the Harris corner detection method.
    This function computes the strength using the determinant and trace of the Hessian matrix 
    components provided.

    Parameters:
    Ix2 (numpy.ndarray): The image derivative squared in the x-direction, weighted by a Gaussian.
    Iy2 (numpy.ndarray): The image derivative squared in the y-direction, weighted by a Gaussian.
    Ixy (numpy.ndarray): The product of the image derivatives in the x and y directions, weighted by a Gaussian.

    Returns:
    numpy.ndarray: An array representing the corner strength at each pixel of the image. High values
                   typically correspond to corner-like features.

    Note:
    This function utilizes the Harris corner response formula: R = Det(H) - k(Trace(H)^2), where H is
    the second moment matrix, Det(H) is the determinant of H, Trace(H) is the trace of H, and k is a
    sensitivity factor, commonly set to 0.1 in literature.
    """

    # Determinant of H matrix
    det_H = Ix2 * Iy2 - Ixy**2

    # Trace of H_matrix
    trace_H = Ix2 + Iy2

    # Corner Strength
    c_H = det_H - 0.1 * (trace_H ** 2)
    return c_H


# Task 3: Orientation at each pixel
def orientation(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:

    """
    Computes the gradient orientation at each pixel of an image based on the x and y gradients.

    Parameters:
    Ix (numpy.ndarray): The image gradient in the x-direction.
    Iy (numpy.ndarray): The image gradient in the y-direction.

    Returns:
    numpy.ndarray: An array representing the gradient orientation at each pixel, measured in radians.
                   The orientation values range from -π to π, describing the angle between the gradient
                   vector and the positive x-axis.

    Note:
    The function uses the arctan2 function, which handles division by zero and provides the correct
    angle quadrant.
    """

    # Angle of gradient
    return np.arctan2(Iy, Ix)


# Task 4: Thresholding
def get_keypoints(c_H: np.ndarray, threshold: float) -> List[Tuple[int, int]]:

    """
    Identifies and returns keypoints in an image by examining the corner strength matrix.
    A keypoint is considered if its corner strength value exceeds a specified threshold and
    it is the maximum value within a 7x7 neighborhood, ensuring it represents a prominent corner.

    Parameters:
    c_H (numpy.ndarray): A corner strength matrix where each value represents the corner strength
                         at that pixel.
    threshold (float): The minimum corner strength value required for a pixel to be considered a keypoint.

    Returns:
    list: A list of tuples, where each tuple (i, j) represents the row and column indices of a keypoint
          in the image.

    Note:
    The function skips the first three and last three rows and columns to avoid border effects where
    the neighborhood might extend beyond the image boundary.
    """

    keypoints = []
    for i in range(3, c_H.shape[0] - 3): # We skip first 3 and last 3 rows to avoid border
        for j in range(3, c_H.shape[1] - 3): # We skip first 3 and last 3 columns to avoid border

            # Get the points who are greater than threshold and maximum in their 7X7 neighborhood
            if c_H[i, j] > threshold and c_H[i, j] == np.max(c_H[i-3:i+4, j-3:j+4]):
                keypoints.append((i, j))

    return keypoints


# Task 5: Display the keypoints
def display_keypoints(image: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:

    """
    Draws keypoints on an image as green circles to visually represent identified keypoints. 
    This is typically used for visualizing features detected in image processing tasks like corner detection.

    Parameters:
    image (numpy.ndarray): The original image on which keypoints will be marked.
    keypoints (list): A list of tuples, where each tuple contains the row (i) and column (j) coordinates
                      of a keypoint.

    Returns:
    numpy.ndarray: The image with keypoints marked in green.

    Note:
    Circles are drawn with a radius of 3 pixels and a thickness of 1 pixel, and can be adjusted as needed.
    """

    output_image = image.copy()  # Make copy of image

    for (i, j) in keypoints:
        cv2.circle(output_image, (j, i), 3, (0, 255, 0), 1)  
        # Draw green circles with a thickness of 1 and radius of 3
        
    return output_image

def detect_keypoint(imagepath: str) -> np.ndarray:
    
    """
    Detects and displays keypoints in an image using Harris Corner Detection. The function reads an image from a given
    path, resizes it, computes the Harris corner response, identifies keypoints based on a threshold, and then visually
    marks these keypoints on the image.

    Parameters:
    imagepath (str): The file path of the image on which keypoints detection will be performed.

    Returns:
    numpy.ndarray: An image with keypoints visually marked, resized to 500x500 pixels.

    Note:
    The threshold for corner detection is set high and may need to be adjusted based on specific image characteristics.
    The function processes images in grayscale for corner detection but displays the keypoints on the original image.
    """

    orig_img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(orig_img, (500, 500))

    # threshold for corner strength
    threshold = 3000000 # Adjust this threshold based on your image

    # Harris Corner Detection
    Ix2, Iy2, Ixy = compute_h_matrix(image)
    c_H = corner_strength(Ix2, Iy2, Ixy)
    keypoints = get_keypoints(c_H, threshold)

    # # Display and save output
    display_image=cv2.imread(imagepath)
    display_image = cv2.resize(display_image, (500, 500))
    keypoint_img=display_keypoints(display_image, keypoints)

    return keypoint_img
