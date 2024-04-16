from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.exceptions import ValidationError
import cv2 # imports OpenCV
from .image_processing import swap_phases, create_hybrid_img, blend, detect_keypoint, detect_edges, blur_image
from .helper import pre_process_img, post_process_img
from django.http import HttpRequest, HttpResponse
from typing import Callable, Dict, Tuple, List, Optional
from django.core.files.uploadedfile import UploadedFile
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Create your views here.

def handle_image_upload(request: HttpRequest, file_keys: List[str] = ['imageFile']) -> Tuple[Optional[UploadedFile], str]:

    """
    Handles image file upload from an HTTP request.

    Parameters:
    request (HttpRequest): The HTTP request containing the file.
    file_keys (list): A list of keys under which image files are uploaded.

    Returns:
    tuple: The first element is the image file if found, or None, and the second element is an error message or empty string.
    """

    if request.method == 'POST' and 'imageFile' in request.FILES:
        image_file = request.FILES['imageFile']
        try:
            return image_file, ''
        except ValidationError as ve:
            return None, str(ve)
       
    return None, "Invalid request or no image found."

def process_and_respond(request: HttpRequest, processing_function: Callable[[np.ndarray, str], np.ndarray]) -> Dict[str, any]:

    """
    Processes an image file from an HTTP request using a provided function and responds with context.

    Parameters:
    request (HttpRequest): The HTTP request object.
    processing_function (function): A function that takes an original image and a file path to process the image.

    Returns:
    dict: A context dictionary with URLs of original and processed images or error messages.
    """

    image_file, error = handle_image_upload(request)
    context = {}
    if image_file:
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            processed_image = processing_function(orig_img, filepath)
            processed_file_url = post_process_img(filepath, processed_image)
            context.update({
                'converted_image_url': processed_file_url,
                'original_image_url': uploaded_file_url
            })
            logger.info(f"Image processed successfully: {processed_file_url}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            context['error_message'] = str(e)
    else:
        logger.warning(f"Failed to upload image: {error}")
        context['error_message'] = error

    return context


def index(request: HttpRequest) -> HttpResponse:

    """
    Renders the main index page.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered index page.
    """

    return render(request, 'index.html')


def convert_rgb_to_bgr(request: HttpRequest) -> HttpResponse:

    """
    Converts an uploaded image from RGB to BGR format and displays both the original and converted images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both the original and the converted image.
    """

    if request.method == 'GET':
        return render(request, 'rgb_to_bgr.html', {})

    def process_image(orig_img, filepath):
        return cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    
    context = process_and_respond(request, process_image)
    return render(request, 'rgb_to_bgr.html', context)


def convert_to_grayscale(request: HttpRequest) -> HttpResponse:

    """
    Converts an uploaded image to grayscale and displays both the original and grayscale images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both the original and the grayscale image.
    """

    if request.method == 'GET':
        return render(request, 'grayscale.html', {})

    def process_image(orig_img, filepath):
        return cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    
    context = process_and_respond(request, process_image)
    return render(request, 'grayscale.html', context)


def resize(request: HttpRequest) -> HttpResponse:

    """
    Resizes an uploaded image based on specified parameters and displays both the original and resized images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both the original and the resized image.
    """

    context = {}
    if request.method == 'POST' and request.FILES['imageFile']:
        image_file = request.FILES['imageFile']
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            ratio = request.POST.get('ratio')
            custom_width = request.POST.get('width')
            custom_height = request.POST.get('height')
            original_height, original_width = orig_img.shape[:2]
            
            logger.debug(f"Resizing parameters received - Ratio: {ratio}, Width: {width}, Height: {height}.")

            if ratio == 'custom' and custom_width and custom_height:
                new_size = (int(custom_width), int(custom_height))  # Resize to custom dimensions
            elif ratio =='custom' and (not custom_width or not custom_height):
                raise ValidationError("Custom height or custom width not given")
            else:
                if ratio:  # Calculate new size based on the selected ratio 
                    new_width = original_width
                    new_height = int(original_width / float(ratio))
                    new_size = (new_width, new_height)
                else:
                    # If no ratio is selected, don't resize
                    new_size = (original_width, original_height)

            processed_image = cv2.resize(orig_img, new_size, interpolation=cv2.INTER_AREA)
            processed_file_url = post_process_img(filepath, processed_image)
            context.update({'converted_image_url': processed_file_url, 'original_image_url': uploaded_file_url})
            logger.info(f"Image resized successfully: {processed_file_url}")
        except ValidationError as ve:
            logger.warning(f"Validation error during resize: {str(ve)}")
            context['error_message'] = str(ve)
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}", exc_info=True)
            context['error_message'] = 'Could Not Process Image'
    return render(request, 'resize_image.html', context)


def swap_phase(request: HttpRequest) -> HttpResponse:
    
    """
    Swaps the Fourier phase components of two uploaded images and displays the original and processed images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page showing the original and phase-swapped images.
    """

    context = {}
    if request.method == 'POST' and request.FILES.get('imageFile1') and request.FILES.get('imageFile2'):
        image_file = request.FILES.get('imageFile1')
        image_file2 = request.FILES.get('imageFile2')
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            uploaded_file_url2, filepath2, orig_img2 = pre_process_img(image_file2)
            swapped_img, swapped_img2 = swap_phases(orig_img, orig_img2)
            processed_file_url = post_process_img(filepath, swapped_img)
            processed_file_url2 = post_process_img(filepath2, swapped_img2)
            context.update({'converted_image_url': processed_file_url, 'original_image_url': uploaded_file_url, 'converted_image_url2': processed_file_url2, 'original_image_url2': uploaded_file_url2})
            logger.info("Phase swap process completed successfully.")
        except ValidationError as ve:
            logger.error(f"Validation error during phase swap: {str(ve)}")
            context['error_message'] = str(ve)
        except Exception as e:
            logger.error(f"Error during phase swap process: {str(e)}", exc_info=True)
            context['error_message'] = 'Could Not Process Image'
    return render(request, 'swap_phase.html', context)


def create_hybrid(request: HttpRequest) -> HttpResponse:

    """
    Creates hybrid images by combining features from two uploaded images and displays them.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to the original and hybrid images.
    """

    context = {}
    if request.method == 'POST' and request.FILES.get('imageFile1') and request.FILES.get('imageFile2'):
        image_file = request.FILES.get('imageFile1')
        image_file2 = request.FILES.get('imageFile2')
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            uploaded_file_url2, filepath2, orig_img2 = pre_process_img(image_file2)
            swapped_img = create_hybrid_img(orig_img,orig_img2)
            swapped_img2 = create_hybrid_img(orig_img2,orig_img)
            processed_file_url = post_process_img(filepath, swapped_img)
            processed_file_url2 = post_process_img(filepath2, swapped_img2)
            context.update({'converted_image_url': processed_file_url, 'original_image_url': uploaded_file_url, 'converted_image_url2': processed_file_url2, 'original_image_url2': uploaded_file_url2})
            logger.info("Hybrid image creation completed successfully.")
        except ValidationError as ve:
            logger.error(f"Validation error during hybrid image creation: {str(ve)}")
            context['error_message'] = str(ve)
        except Exception as e:
            logger.error(f"Error during hybrid image creation: {str(e)}", exc_info=True)
            context['error_message'] = 'Could Not Process Image'
    return render(request, 'hybrid.html', context)


def blend_images(request: HttpRequest) -> HttpResponse:

    """
    Blends two uploaded images and displays the original and blended images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both original images and the blended image.
    """

    context = {}
    if request.method == 'POST' and request.FILES.get('imageFile1') and request.FILES.get('imageFile2'):
        image_file = request.FILES.get('imageFile1')
        image_file2 = request.FILES.get('imageFile2')
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            uploaded_file_url2, filepath2, orig_img2 = pre_process_img(image_file2)
            blended_img = blend(orig_img,orig_img2)
            processed_file_url = post_process_img(filepath, blended_img)
            context.update({'converted_image_url': processed_file_url, 'original_image_url': uploaded_file_url, 'original_image_url2':uploaded_file_url2})
            logger.info("Image blending completed successfully.")
        except ValidationError as ve:
            logger.error(f"Validation error during image blending: {str(ve)}")
            context['error_message'] = str(ve)
        except Exception as e:
            logger.error(f"Error during image blending: {str(e)}", exc_info=True)
            context['error_message'] = 'Could Not Process Image'
    return render(request, 'blend_img.html', context)


def detect_keypoint_in_img(request: HttpRequest) -> HttpResponse:

    """
    Detects keypoints in an uploaded image using Harris corner detection and displays the keypoints.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with the original image and its keypoints highlighted.
    """

    if request.method == 'GET':
        return render(request, 'keypoints.html', {})

    def process_image(orig_img, filepath):
        return detect_keypoint(filepath)
    
    context = process_and_respond(request, process_image)
    return render(request, 'keypoints.html', context)


def crop_image(request: HttpRequest) -> HttpResponse:

    """
    Crops an uploaded image based on user-specified coordinates and displays both the original and cropped images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both the original and the cropped image.
    """

    context={}
    if request.method == 'POST' and request.FILES['imageFile']:
        image_file = request.FILES['imageFile']
        try:
            uploaded_file_url, filepath, orig_img = pre_process_img(image_file)
            x = int(float(request.POST['cropX']))
            y = int(float(request.POST['cropY']))
            width = int(float(request.POST['cropWidth']))
            height = int(float(request.POST['cropHeight']))
            x2 = x + width
            y2 = y + height
            processed_image = orig_img[y:y2, x:x2]
            processed_file_url = post_process_img(filepath, processed_image)
            context.update({'converted_image_url': processed_file_url, 'original_image_url': uploaded_file_url})
        except ValidationError as ve:
            context['error_message'] = str(ve)
        except Exception as e:
            context['error_message'] = 'Could Not Process Image'
    return render(request, 'crop.html', context)


def detect_edge(request: HttpRequest) -> HttpResponse:
    """
    Detects edges in an uploaded image using a Sobel filter and displays both the original and edge-detected images.

    Parameters:
    request (HttpRequest): The HTTP request object.

    Returns:
    HttpResponse: The rendered page with links to both the original and the edge-detected image.
    """

    if request.method == 'GET':
        return render(request, 'detect_edge.html', {})

    def process_image(orig_img, filepath):
        return detect_edges(orig_img)
    
    context = process_and_respond(request, process_image)
    return render(request, 'detect_edge.html', context)
    

# def blur_img(request):
#     context={}
#     if request.method == 'POST' and request.FILES['imageFile']:
        
#         image_file = request.FILES['imageFile']

#         try:
#             # Validate file
#             validate_image(image_file)

#             uploaded_file_url, filepath=save_img(image_file)

#             orig_img = cv2.imread(filepath)
#             processed_image = blur_image(orig_img)
        
#             processed_filename = change_filename(filepath)
#             processed_file_url=cv2_write(processed_filename, processed_image)
            
#             context['converted_image_url'] = processed_file_url
#             context['original_image_url'] = uploaded_file_url

#         except ValidationError as ve:
#             context['error_message'] = str(ve)
#         except Exception as e:
#             context['error_message'] = 'Could Not Process Image'

#     return render(request, 'blur_img.html', context)

