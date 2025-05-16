import cv2
import numpy as np

def enhance_thermal_image_guided_filtering(image, num_scales=3, detail_enhancement_factors=None):
    """
    Enhance thermal image using Multi-Scale Guided Filtering with CLAHE.
    This algorithm is particularly effective for enhancing thermal facial features
    related to alcohol consumption.
    
    Args:
        image: Input grayscale image
        num_scales: Number of scales for decomposition
        detail_enhancement_factors: List of enhancement factors for each scale
        
    Returns:
        Enhanced image
    """
    # Default enhancement factors if not provided
    if detail_enhancement_factors is None:
        detail_enhancement_factors = [1.5, 1.8, 2.0]
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Multi-scale decomposition
    base_layer = image_float
    detail_layers = []
    
    for i in range(num_scales):
        radius = 2**(i+1)
        epsilon = 0.1**2
        
        # Apply guided filter
        filtered_image = cv2.ximgproc.guidedFilter(
            guide=base_layer, 
            src=base_layer, 
            radius=radius, 
            eps=epsilon
        )
        
        # Extract detail layer
        detail_layer = base_layer - filtered_image
        detail_layers.append(detail_layer)
        
        # Update base layer
        base_layer = filtered_image
    
    # Enhance base layer with CLAHE
    base_uint8 = np.clip(base_layer * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_base = clahe.apply(base_uint8).astype(np.float32) / 255.0
    
    # Enhance detail layers
    enhanced_details = []
    for i in range(num_scales):
        enhanced_detail = detail_layers[i] * detail_enhancement_factors[i]
        enhanced_details.append(enhanced_detail)
    
    # Reconstruct enhanced image
    result = enhanced_base
    for detail in enhanced_details:
        result = result + detail
    
    # Normalize to valid range
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

def enhance_thermal_image_tophat(image, num_scales=4, bright_weight=1.2, dark_weight=0.8):
    """
    Enhance thermal image using Multiscale Top-Hat Transform.
    This algorithm is effective at highlighting both bright and dark thermal regions,
    making it suitable for detecting alcohol-related facial features.
    
    Args:
        image: Input grayscale image
        num_scales: Number of scales for structuring elements
        bright_weight: Weight for bright features
        dark_weight: Weight for dark features
        
    Returns:
        Enhanced image
    """
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    # Initialize structuring elements
    structuring_elements = []
    for i in range(num_scales):
        size = 2*i + 3  # Odd sizes: 3, 5, 7, 9
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        structuring_elements.append(se)
    
    # Extract bright features (White Top-Hat)
    bright_features = []
    for se in structuring_elements:
        opening = cv2.morphologyEx(image_float, cv2.MORPH_OPEN, se)
        white_top_hat = image_float - opening
        bright_features.append(white_top_hat)
    
    # Extract dark features (Black Top-Hat)
    dark_features = []
    for se in structuring_elements:
        closing = cv2.morphologyEx(image_float, cv2.MORPH_CLOSE, se)
        black_top_hat = closing - image_float
        dark_features.append(black_top_hat)
    
    # Combine features from all scales
    combined_bright = np.sum(bright_features, axis=0)
    combined_dark = np.sum(dark_features, axis=0)
    
    # Enhance original image
    result = image_float + (bright_weight * combined_bright) - (dark_weight * combined_dark)
    
    # Normalize to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def enhance_thermal_image_alcohol_detection(image, face_cascade_path=None):
    """
    Region-specific enhancement for alcohol detection in thermal facial images.
    This algorithm targets the specific facial regions known to exhibit thermal changes
    after alcohol consumption: the forehead (cooling) and nose (warming).
    
    Args:
        image: Input grayscale image
        face_cascade_path: Path to Haar cascade XML file for face detection
        
    Returns:
        Enhanced image with region-specific processing
    """
    # If no face cascade provided, use a simple approach without face detection
    if face_cascade_path is None:
        # Estimate forehead region (upper third of image)
        h, w = image.shape[:2]
        forehead_region = image[:h//3, :]
        
        # Estimate nose region (center of image)
        center_y, center_x = h//2, w//2
        nose_size = min(h, w) // 5
        nose_region = image[
            max(0, center_y-nose_size):min(h, center_y+nose_size),
            max(0, center_x-nose_size):min(w, center_x+nose_size)
        ]
        
        # Create mask for the rest of the image
        mask = np.ones_like(image, dtype=np.uint8) * 255
        mask[:h//3, :] = 0  # Remove forehead region
        mask[
            max(0, center_y-nose_size):min(h, center_y+nose_size),
            max(0, center_x-nose_size):min(w, center_x+nose_size)
        ] = 0  # Remove nose region
        other_regions = cv2.bitwise_and(image, image, mask=mask)
    else:
        # Use face detection for more accurate region extraction
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        
        if len(faces) == 0:
            # Fall back to simple approach if no face detected
            return enhance_thermal_image_alcohol_detection(image)
        
        # Use the first detected face
        x, y, w, h = faces[0]
        
        # Extract forehead region (upper third of face)
        forehead_region = image[y:y+h//3, x:x+w]
        
        # Extract nose region (center of face)
        nose_y = y + h//2
        nose_x = x + w//2
        nose_size = min(h, w) // 5
        nose_region = image[
            max(0, nose_y-nose_size):min(image.shape[0], nose_y+nose_size),
            max(0, nose_x-nose_size):min(image.shape[1], nose_x+nose_size)
        ]
        
        # Create mask for the rest of the image
        mask = np.ones_like(image, dtype=np.uint8) * 255
        mask[y:y+h//3, x:x+w] = 0  # Remove forehead region
        mask[
            max(0, nose_y-nose_size):min(image.shape[0], nose_y+nose_size),
            max(0, nose_x-nose_size):min(image.shape[1], nose_x+nose_size)
        ] = 0  # Remove nose region
        other_regions = cv2.bitwise_and(image, image, mask=mask)
    
    # Enhance forehead region (cooling effect) - increase contrast in cooler areas
    # Use CLAHE with higher clip limit
    clahe_forehead = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_forehead = clahe_forehead.apply(forehead_region)
    
    # Apply multiscale top-hat with emphasis on dark features
    enhanced_forehead = enhance_thermal_image_tophat(
        enhanced_forehead, 
        num_scales=3, 
        bright_weight=0.8, 
        dark_weight=1.5
    )
    
    # Enhance nose region (warming effect) - increase contrast in warmer areas
    # Use CLAHE with moderate clip limit
    clahe_nose = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_nose = clahe_nose.apply(nose_region)
    
    # Apply multiscale top-hat with emphasis on bright features
    enhanced_nose = enhance_thermal_image_tophat(
        enhanced_nose, 
        num_scales=3, 
        bright_weight=1.5, 
        dark_weight=0.8
    )
    
    # Apply general enhancement to other regions
    clahe_general = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_other = clahe_general.apply(other_regions)
    
    # Create result image
    result = image.copy()
    
    # If no face cascade provided, use the simple approach
    if face_cascade_path is None:
        # Copy enhanced regions back to result
        result[:h//3, :] = enhanced_forehead
        result[
            max(0, center_y-nose_size):min(h, center_y+nose_size),
            max(0, center_x-nose_size):min(w, center_x+nose_size)
        ] = enhanced_nose
        
        # Add other regions
        other_mask = mask > 0
        result[other_mask] = enhanced_other[other_mask]
    else:
        # Use face detection results
        if len(faces) > 0:
            x, y, w, h = faces[0]
            
            # Copy enhanced forehead
            result[y:y+h//3, x:x+w] = enhanced_forehead
            
            # Copy enhanced nose
            nose_y = y + h//2
            nose_x = x + w//2
            nose_size = min(h, w) // 5
            result[
                max(0, nose_y-nose_size):min(image.shape[0], nose_y+nose_size),
                max(0, nose_x-nose_size):min(image.shape[1], nose_x+nose_size)
            ] = enhanced_nose
            
            # Add other regions
            other_mask = mask > 0
            result[other_mask] = enhanced_other[other_mask]
    
    return result

def combined_enhancement_for_alcohol_detection(image):
    """
    Combined approach using multiple enhancement techniques for optimal
    detection of alcohol-related thermal features.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Enhanced image
    """
    # Step 1: Apply guided filtering for overall enhancement
    guided_enhanced = enhance_thermal_image_guided_filtering(image)
    
    # Step 2: Apply region-specific enhancement for alcohol detection
    result = enhance_thermal_image_alcohol_detection(guided_enhanced)
    
    # Step 3: Final contrast adjustment
    clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe_final.apply(result)
    
    return result
