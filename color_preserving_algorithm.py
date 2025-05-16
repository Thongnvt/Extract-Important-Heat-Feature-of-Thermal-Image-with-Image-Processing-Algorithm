import cv2
import numpy as np

def apply_colormap_to_enhanced_image(original_color_img, enhanced_grayscale_img):
    """
    Apply the color mapping from the original thermal image to the enhanced grayscale image.
    
    Args:
        original_color_img: Original thermal image with color spectrum
        enhanced_grayscale_img: Enhanced grayscale image
        
    Returns:
        Enhanced image with original color spectrum preserved
    """
    # Convert original image to HSV color space
    original_hsv = cv2.cvtColor(original_color_img, cv2.COLOR_BGR2HSV)
    
    # Normalize the enhanced grayscale image to 0-255 range if needed
    if enhanced_grayscale_img.dtype != np.uint8:
        enhanced_normalized = cv2.normalize(enhanced_grayscale_img, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_uint8 = enhanced_normalized.astype(np.uint8)
    else:
        enhanced_uint8 = enhanced_grayscale_img
    
    # Create a new HSV image using:
    # - Hue from the original image
    # - Saturation from the original image
    # - Value (brightness) from the enhanced image
    new_hsv = original_hsv.copy()
    new_hsv[:,:,2] = enhanced_uint8
    
    # Convert back to BGR color space
    enhanced_color_img = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced_color_img

def enhance_thermal_image_guided_filtering_color(image, original_color_img, num_scales=3, detail_enhancement_factors=None):
    """
    Enhance thermal image using Multi-Scale Guided Filtering with CLAHE,
    preserving the original color spectrum.
    
    Args:
        image: Input grayscale image
        original_color_img: Original thermal image with color spectrum
        num_scales: Number of scales for decomposition
        detail_enhancement_factors: List of enhancement factors for each scale
        
    Returns:
        Enhanced image with original color spectrum preserved
    """
    # First apply the grayscale enhancement
    enhanced_gray = enhance_thermal_image_guided_filtering(image, num_scales, detail_enhancement_factors)
    
    # Apply the original color mapping to the enhanced image
    enhanced_color = apply_colormap_to_enhanced_image(original_color_img, enhanced_gray)
    
    return enhanced_color

def enhance_thermal_image_tophat_color(image, original_color_img, num_scales=4, bright_weight=1.2, dark_weight=0.8):
    """
    Enhance thermal image using Multiscale Top-Hat Transform,
    preserving the original color spectrum.
    
    Args:
        image: Input grayscale image
        original_color_img: Original thermal image with color spectrum
        num_scales: Number of scales for structuring elements
        bright_weight: Weight for bright features
        dark_weight: Weight for dark features
        
    Returns:
        Enhanced image with original color spectrum preserved
    """
    # First apply the grayscale enhancement
    enhanced_gray = enhance_thermal_image_tophat(image, num_scales, bright_weight, dark_weight)
    
    # Apply the original color mapping to the enhanced image
    enhanced_color = apply_colormap_to_enhanced_image(original_color_img, enhanced_gray)
    
    return enhanced_color

def enhance_thermal_image_alcohol_detection_color(image, original_color_img, face_cascade_path=None):
    """
    Region-specific enhancement for alcohol detection in thermal facial images,
    preserving the original color spectrum.
    
    Args:
        image: Input grayscale image
        original_color_img: Original thermal image with color spectrum
        face_cascade_path: Path to Haar cascade XML file for face detection
        
    Returns:
        Enhanced image with original color spectrum preserved
    """
    # First apply the grayscale enhancement
    enhanced_gray = enhance_thermal_image_alcohol_detection(image, face_cascade_path)
    
    # Apply the original color mapping to the enhanced image
    enhanced_color = apply_colormap_to_enhanced_image(original_color_img, enhanced_gray)
    
    return enhanced_color

def combined_enhancement_for_alcohol_detection_color(image, original_color_img):
    """
    Combined approach using multiple enhancement techniques for optimal
    detection of alcohol-related thermal features, preserving the original color spectrum.
    
    Args:
        image: Input grayscale image
        original_color_img: Original thermal image with color spectrum
        
    Returns:
        Enhanced image with original color spectrum preserved
    """
    # First apply the grayscale enhancement
    enhanced_gray = combined_enhancement_for_alcohol_detection(image)
    
    # Apply the original color mapping to the enhanced image
    enhanced_color = apply_colormap_to_enhanced_image(original_color_img, enhanced_gray)
    
    return enhanced_color

# Import the original grayscale enhancement functions
from thermal_algorithms import (
    enhance_thermal_image_guided_filtering,
    enhance_thermal_image_tophat,
    enhance_thermal_image_alcohol_detection,
    combined_enhancement_for_alcohol_detection
)
