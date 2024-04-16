from django.urls import path
from .views import index, convert_rgb_to_bgr, convert_to_grayscale, resize, swap_phase
from .views import create_hybrid, blend_images, detect_keypoint_in_img, crop_image, detect_edge

urlpatterns = [
    path('', index, name='home'),
    path('rgb-to-bgr/', convert_rgb_to_bgr, name='rgb_to_bgr'),
    path('rgb-to-grayscale/', convert_to_grayscale, name='convert_to_grayscale'),
    path('resize/', resize, name='resize'),
    path('swap_phase/', swap_phase, name='swap_phase'),
    path('hybrid/', create_hybrid, name='hybrid'),
    path('blending/', blend_images, name='blend'),
    path('keypoint/', detect_keypoint_in_img, name='keypoint'),
    path('crop/', crop_image, name='crop'),
    path('detect_edge/', detect_edge, name='edge'),
    # path('image_blur/', blur_img, name='blur'),
]
