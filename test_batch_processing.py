import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_batch_processing():
    """
    Test the batch processing functionality with a sample image.
    This function creates a test image, processes it with different methods,
    and verifies the results.
    """
    # Create test directories
    test_input_dir = "test_input"
    test_output_dir = "test_output"
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create a sample thermal image (gradient from blue to red)
    width, height = 400, 300
    sample_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a temperature gradient (simulating thermal image)
    for y in range(height):
        for x in range(width):
            # Create a radial gradient
            cx, cy = width // 2, height // 2
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_distance = np.sqrt(cx**2 + cy**2)
            normalized_distance = distance / max_distance
            
            # Map distance to color (blue -> green -> yellow -> red)
            if normalized_distance < 0.25:
                # Blue
                sample_image[y, x] = [255, 0, 0]
            elif normalized_distance < 0.5:
                # Green
                sample_image[y, x] = [0, 255, 0]
            elif normalized_distance < 0.75:
                # Yellow
                sample_image[y, x] = [0, 255, 255]
            else:
                # Red
                sample_image[y, x] = [0, 0, 255]
    
    # Add some facial-like features
    # Eyes
    cv2.circle(sample_image, (width//2 - 50, height//2 - 30), 20, (0, 0, 255), -1)
    cv2.circle(sample_image, (width//2 + 50, height//2 - 30), 20, (0, 0, 255), -1)
    # Nose
    cv2.rectangle(sample_image, (width//2 - 20, height//2), (width//2 + 20, height//2 + 40), (0, 255, 255), -1)
    # Mouth
    cv2.ellipse(sample_image, (width//2, height//2 + 70), (60, 20), 0, 0, 180, (0, 255, 0), -1)
    
    # Save the sample image
    sample_image_path = os.path.join(test_input_dir, "sample_thermal.png")
    cv2.imwrite(sample_image_path, sample_image)
    print(f"Created sample thermal image: {sample_image_path}")
    
    # Test batch processing with different methods
    methods = [1, 2, 3, 4, 5]
    color_options = [False, True]
    
    for method in methods:
        for color in color_options:
            # Build the command
            color_flag = "--color" if color else ""
            method_name = {
                1: "CLAHE",
                2: "Multi-Scale Guided Filtering",
                3: "Multiscale Top-Hat Transform",
                4: "Region-Specific Alcohol Detection",
                5: "Combined Approach"
            }.get(method, "Unknown")
            
            print(f"\nTesting Method {method}: {method_name} (Color: {color})")
            cmd = f"python batch_processing.py --input {test_input_dir} --output {test_output_dir} --method {method} {color_flag}"
            print(f"Running: {cmd}")
            
            # Execute the command
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"Error: Command failed with exit code {exit_code}")
                continue
            
            # Check if output file exists
            color_suffix = "_color" if color else ""
            expected_output = os.path.join(
                test_output_dir, 
                f"sample_thermal_{method_name.replace(' ', '')}{color_suffix}.png"
            )
            
            if os.path.exists(expected_output):
                print(f"Success: Output file created at {expected_output}")
            else:
                print(f"Error: Output file not found at {expected_output}")
                # List files in output directory
                print("Files in output directory:")
                for file in os.listdir(test_output_dir):
                    print(f"  {file}")
    
    # Display results
    print("\nTest completed. Check the test_output directory for results.")
    
    # Create a visualization of all results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(3, 4, 1)
    original = cv2.imread(sample_image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis('off')
    
    # Plot all results
    plot_idx = 2
    for method in methods:
        method_name = {
            1: "CLAHE",
            2: "MultiScaleGuidedFilter",
            3: "MultiscaleTopHat",
            4: "RegionSpecificAlcohol",
            5: "CombinedApproach"
        }.get(method, "Unknown")
        
        # Grayscale version
        output_path = os.path.join(test_output_dir, f"sample_thermal_{method_name}.png")
        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 4, plot_idx)
            plt.imshow(img_rgb)
            plt.title(f"{method_name}\n(Grayscale)")
            plt.axis('off')
        else:
            plt.subplot(3, 4, plot_idx)
            plt.text(0.5, 0.5, "Not Found", ha='center', va='center')
            plt.title(f"{method_name}\n(Grayscale)")
            plt.axis('off')
        plot_idx += 1
        
        # Color version
        output_path = os.path.join(test_output_dir, f"sample_thermal_{method_name}_color.png")
        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 4, plot_idx + 4)
            plt.imshow(img_rgb)
            plt.title(f"{method_name}\n(Color)")
            plt.axis('off')
        else:
            plt.subplot(3, 4, plot_idx + 4)
            plt.text(0.5, 0.5, "Not Found", ha='center', va='center')
            plt.title(f"{method_name}\n(Color)")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_output_dir, "batch_processing_test_results.png"))
    print(f"Results visualization saved to {os.path.join(test_output_dir, 'batch_processing_test_results.png')}")

if __name__ == "__main__":
    test_batch_processing()
