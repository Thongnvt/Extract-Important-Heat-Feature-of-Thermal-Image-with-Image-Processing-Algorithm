import cv2
import os
import numpy as np
import argparse
import glob
from tqdm import tqdm

# Import color-preserving thermal algorithms
try:
    from color_preserving_algorithm import (
        enhance_thermal_image_guided_filtering_color,
        enhance_thermal_image_tophat_color,
        enhance_thermal_image_alcohol_detection_color,
        combined_enhancement_for_alcohol_detection_color
    )
except ImportError:
    print("Warning: color_preserving_algorithms.py not found. Color preservation will not be available.")
    color_preservation_available = False
else:
    color_preservation_available = True

# Import original grayscale algorithms for processing
try:
    from thermal_algorithms import (
        enhance_thermal_image_guided_filtering,
        enhance_thermal_image_tophat,
        enhance_thermal_image_alcohol_detection,
        combined_enhancement_for_alcohol_detection
    )
except ImportError:
    print("Error: enhanced_thermal_algorithms.py not found. This file is required.")
    exit(1)


def apply_clahe(gray_img, clip_limit=1.0, tile_grid_size=(8, 8)):
    """Apply CLAHE enhancement to grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def apply_clahe_color(gray_img, original_img, clip_limit=1.0, tile_grid_size=(8, 8)):
    """Apply CLAHE enhancement while preserving original color spectrum."""
    # Apply CLAHE to grayscale image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_gray = clahe.apply(gray_img)
    
    # Convert original image to HSV
    original_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    
    # Create a new HSV image using hue and saturation from original, value from enhanced
    new_hsv = original_hsv.copy()
    new_hsv[:,:,2] = enhanced_gray
    
    # Convert back to BGR
    enhanced_color = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced_color


def process_image(image_path, enhancement_method, preserve_color, output_dir):
    """
    Process a single thermal image with the specified enhancement method.
    
    Args:
        image_path: Path to the input image
        enhancement_method: Integer representing the enhancement method (1-5)
        preserve_color: Boolean indicating whether to preserve color
        output_dir: Directory to save the enhanced image
        
    Returns:
        Path to the saved enhanced image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply the selected enhancement method
    if enhancement_method == 1:  # CLAHE
        if preserve_color and color_preservation_available:
            enhanced_img = apply_clahe_color(gray_img, img)
        else:
            enhanced_img = apply_clahe(gray_img)
    
    elif enhancement_method == 2:  # Multi-Scale Guided Filtering
        if preserve_color and color_preservation_available:
            enhanced_img = enhance_thermal_image_guided_filtering_color(gray_img, img)
        else:
            enhanced_img = enhance_thermal_image_guided_filtering(gray_img)
    
    elif enhancement_method == 3:  # Multiscale Top-Hat Transform
        if preserve_color and color_preservation_available:
            enhanced_img = enhance_thermal_image_tophat_color(gray_img, img)
        else:
            enhanced_img = enhance_thermal_image_tophat(gray_img)
    
    elif enhancement_method == 4:  # Region-Specific Alcohol Detection
        if preserve_color and color_preservation_available:
            enhanced_img = enhance_thermal_image_alcohol_detection_color(gray_img, img)
        else:
            enhanced_img = enhance_thermal_image_alcohol_detection(gray_img)
    
    elif enhancement_method == 5:  # Combined Approach
        if preserve_color and color_preservation_available:
            enhanced_img = combined_enhancement_for_alcohol_detection_color(gray_img, img)
        else:
            enhanced_img = combined_enhancement_for_alcohol_detection(gray_img)
    
    else:
        print(f"Error: Invalid enhancement method {enhancement_method}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the filename without path
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    
    # Get enhancement method name for the output filename
    enhancement_names = {
        1: "CLAHE",
        2: "MultiScaleGuidedFilter",
        3: "MultiscaleTopHat",
        4: "RegionSpecificAlcohol",
        5: "CombinedApproach"
    }
    enhancement_name = enhancement_names.get(enhancement_method, "Enhanced")
    color_suffix = "_color" if preserve_color else ""
    
    # Create output filename
    output_filename = f"{base_name}_{enhancement_name}{color_suffix}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_img)
    
    return output_path


def process_folder(input_folder, enhancement_method, preserve_color, output_folder):
    """
    Process all images in a folder with the specified enhancement method.
    
    Args:
        input_folder: Path to the input folder containing images
        enhancement_method: Integer representing the enhancement method (1-5)
        preserve_color: Boolean indicating whether to preserve color
        output_folder: Directory to save the enhanced images
        
    Returns:
        List of paths to the saved enhanced images
    """
    # Get all image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"Error: No image files found in {input_folder}")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image
    output_paths = []
    print(f"Processing {len(image_files)} images...")
    for image_path in tqdm(image_files):
        output_path = process_image(image_path, enhancement_method, preserve_color, output_folder)
        if output_path:
            output_paths.append(output_path)
    
    return output_paths


def main():
    """Main function to run the batch thermal image enhancement."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch Thermal Image Enhancement')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder containing thermal images')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for enhanced images')
    parser.add_argument('--method', '-m', type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help='Enhancement method: 1=CLAHE, 2=Multi-Scale Guided Filtering, '
                             '3=Multiscale Top-Hat Transform, 4=Region-Specific Alcohol Detection, '
                             '5=Combined Approach')
    parser.add_argument('--color', '-c', action='store_true',
                        help='Preserve original thermal color spectrum')
    
    args = parser.parse_args()
    
    # Check if color preservation is requested but not available
    if args.color and not color_preservation_available:
        print("Warning: Color preservation requested but color_preserving_algorithms.py not found.")
        print("Falling back to grayscale enhancement.")
        args.color = False
    
    # Process the folder
    output_paths = process_folder(args.input, args.method, args.color, args.output)
    
    # Print summary
    if output_paths:
        print(f"Successfully processed {len(output_paths)} images.")
        print(f"Enhanced images saved to {args.output}")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    main()
