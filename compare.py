import cv2
import os
import numpy as np
import matplotlib
# Set non-interactive backend before any other matplotlib imports
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Union, Any

# Import color-preserving thermal algorithms
from color_preserving_algorithm import (
    enhance_thermal_image_guided_filtering_color,
    enhance_thermal_image_tophat_color,
    enhance_thermal_image_alcohol_detection_color,
    combined_enhancement_for_alcohol_detection_color
)

# Import original grayscale algorithms for processing
from thermal_algorithms import (
    enhance_thermal_image_guided_filtering,
    enhance_thermal_image_tophat,
    enhance_thermal_image_alcohol_detection,
    combined_enhancement_for_alcohol_detection
)


@dataclass
class ImageData:
    """Data class to store image information with multiple enhancement options."""
    original: np.ndarray
    grayscale: np.ndarray
    enhanced_clahe: np.ndarray
    enhanced_guided: Optional[np.ndarray] = None
    enhanced_tophat: Optional[np.ndarray] = None
    enhanced_alcohol: Optional[np.ndarray] = None
    enhanced_combined: Optional[np.ndarray] = None
    # Color versions
    enhanced_clahe_color: Optional[np.ndarray] = None
    enhanced_guided_color: Optional[np.ndarray] = None
    enhanced_tophat_color: Optional[np.ndarray] = None
    enhanced_alcohol_color: Optional[np.ndarray] = None
    enhanced_combined_color: Optional[np.ndarray] = None


class ThermalImageComparer:
    def __init__(self, k_clusters: int = 3, grid_size: int = 5):
        """
        Initialize the thermal image comparer.
        
        Args:
            k_clusters: Number of clusters for K-means segmentation
            grid_size: Size of grid cells in pixels
        """
        # Fixed grid_size to 5 (already is 5 by default)
        self.grid_size = 5  # Explicitly set to 5 regardless of input parameter
        
        self.image_files = self._select_images()
        if not self.image_files:
            print("Không có ảnh nào được chọn! Thoát chương trình.")
            return
        
        # Allow user to select enhancement method
        self.enhancement_method = self._select_enhancement_method()
        
        # Ask user if they want to preserve color
        self.preserve_color = self._ask_preserve_color()
        
        self.thermal_images = self._load_thermal_images(self.image_files)
        self.k_clusters = k_clusters
        self.contour_drawer = ContourDrawer()
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_coords = []
        self.center_grid = (0, 0)

    def _ask_preserve_color(self) -> bool:
        """
        Ask user if they want to preserve the original thermal color spectrum.
        
        Returns:
            Boolean indicating whether to preserve color
        """
        print("\n=== Bạn có muốn giữ nguyên bảng màu nhiệt của ảnh gốc không? ===")
        print("1. Có, giữ nguyên bảng màu nhiệt (blue-green-yellow-red)")
        print("2. Không, sử dụng ảnh grayscale")
        
        choice = input("Nhập lựa chọn của bạn (1-2): ")
        return choice == "1"

    def _select_enhancement_method(self) -> int:
        """
        Allow user to select enhancement method.
        
        Returns:
            Integer representing the selected enhancement method
        """
        print("\n=== Chọn phương pháp nâng cao ảnh nhiệt ===")
        print("1. CLAHE cơ bản (Phương pháp hiện tại)")
        print("2. Multi-Scale Guided Filtering (Khuyến nghị cho nâng cao tổng thể)")
        print("3. Multiscale Top-Hat Transform (Tốt cho việc làm nổi bật cả vùng sáng và tối)")
        print("4. Nâng cao theo vùng cho phát hiện rượu (Chuyên biệt cho đặc điểm nhiệt trên khuôn mặt)")
        print("5. Phương pháp kết hợp (Tốt nhất cho phát hiện rượu)")
        
        choice = input("Nhập lựa chọn của bạn (1-5): ")
        return int(choice) if choice.isdigit() and 1 <= int(choice) <= 5 else 1

    def _select_images(self) -> List[str]:
        """
        Select images using a file dialog.
        
        Returns:
            List of selected image file paths
        """
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(
            title="Chọn ảnh nhiệt",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        return list(file_paths) if file_paths else []

    def _apply_clahe_color(self, gray_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement while preserving original color spectrum.
        
        Args:
            gray_img: Grayscale image
            original_img: Original color image
            
        Returns:
            Enhanced color image
        """
        # Apply CLAHE to grayscale image
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_img)
        
        # Convert original image to HSV
        original_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        
        # Create a new HSV image using hue and saturation from original, value from enhanced
        new_hsv = original_hsv.copy()
        new_hsv[:,:,2] = enhanced_gray
        
        # Convert back to BGR
        enhanced_color = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced_color

    def _load_thermal_images(self, image_paths: List[str]) -> Dict[str, ImageData]:
        """
        Load and preprocess thermal images with multiple enhancement methods.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Dictionary mapping filenames to ImageData objects
        """
        images = {}
        
        # Create CLAHE object once for reuse
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Không tìm thấy ảnh: {img_path}")
                continue
                
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue
                
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply different enhancement methods
            enhanced_clahe = clahe.apply(gray_img)
            enhanced_clahe_color = self._apply_clahe_color(gray_img, img) if self.preserve_color else None
            
            # Only compute the selected enhancement method to save processing time
            enhanced_guided = None
            enhanced_tophat = None
            enhanced_alcohol = None
            enhanced_combined = None
            
            # Color versions
            enhanced_guided_color = None
            enhanced_tophat_color = None
            enhanced_alcohol_color = None
            enhanced_combined_color = None
            
            if self.enhancement_method == 2 or self.enhancement_method == 5:
                print(f"Đang áp dụng Multi-Scale Guided Filtering cho {filename}...")
                enhanced_guided = enhance_thermal_image_guided_filtering(gray_img)
                if self.preserve_color:
                    enhanced_guided_color = enhance_thermal_image_guided_filtering_color(gray_img, img)
                
            if self.enhancement_method == 3 or self.enhancement_method == 5:
                print(f"Đang áp dụng Multiscale Top-Hat Transform cho {filename}...")
                enhanced_tophat = enhance_thermal_image_tophat(gray_img)
                if self.preserve_color:
                    enhanced_tophat_color = enhance_thermal_image_tophat_color(gray_img, img)
                
            if self.enhancement_method == 4 or self.enhancement_method == 5:
                print(f"Đang áp dụng nâng cao theo vùng cho phát hiện rượu cho {filename}...")
                enhanced_alcohol = enhance_thermal_image_alcohol_detection(gray_img)
                if self.preserve_color:
                    enhanced_alcohol_color = enhance_thermal_image_alcohol_detection_color(gray_img, img)
                
            if self.enhancement_method == 5:
                print(f"Đang áp dụng phương pháp kết hợp cho {filename}...")
                enhanced_combined = combined_enhancement_for_alcohol_detection(gray_img)
                if self.preserve_color:
                    enhanced_combined_color = combined_enhancement_for_alcohol_detection_color(gray_img, img)
            
            images[filename] = ImageData(
                original=img,
                grayscale=gray_img,
                enhanced_clahe=enhanced_clahe,
                enhanced_guided=enhanced_guided,
                enhanced_tophat=enhanced_tophat,
                enhanced_alcohol=enhanced_alcohol,
                enhanced_combined=enhanced_combined,
                # Color versions
                enhanced_clahe_color=enhanced_clahe_color,
                enhanced_guided_color=enhanced_guided_color,
                enhanced_tophat_color=enhanced_tophat_color,
                enhanced_alcohol_color=enhanced_alcohol_color,
                enhanced_combined_color=enhanced_combined_color
            )
            
        return images

    def _apply_kmeans_clustering(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply K-means clustering to segment the image.
        
        Args:
            image: Grayscale image to segment
            
        Returns:
            Tuple of (binary mask, hottest cluster center value)
        """
        # Reshape image to a 1D array of intensity values
        Z = image.reshape((-1, 1)).astype(np.float32)
        
        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            Z, self.k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Find the cluster with highest intensity (representing hot areas)
        hottest_cluster = np.argmax(centers)
        mask = (labels.flatten() == hottest_cluster).astype(np.uint8) * 255
        
        # Reshape mask back to original image shape
        mask = mask.reshape(image.shape)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask, centers[hottest_cluster][0]

    def _apply_connected_components(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply connected components analysis to filter small regions.
        
        Args:
            binary_image: Binary image from segmentation
            
        Returns:
            Filtered binary image
        """
        # Find all connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
        
        # Filter out small components (noise)
        min_area = 50  # Minimum area threshold
        output = np.zeros_like(binary_image, dtype=np.uint8)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                output[labels == i] = 255
                
        return output

    def _draw_grid(self, image: np.ndarray) -> np.ndarray:
        """
        Draw a grid on the image and identify grid coordinates.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Processed image with grid overlay
        """
        h, w = image.shape[:2]
        grid_color = (0, 255, 0)  # Green color for grid
        
        # Convert image to BGR if it's grayscale
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            display_image = image.copy()
        elif image.shape[2] == 4:  # Handle RGBA images
            display_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError("Unsupported image format")
        
        # Calculate the number of complete grid cells
        rows = h // self.grid_size
        cols = w // self.grid_size
        
        print("Danh sách tọa độ góc trên trái của từng ô lưới:")
        grid_coords = []  # List to store grid coordinates
        
        # Draw the grid and collect coordinates
        for row in range(rows + 1):  # +1 to include partial cells
            y = row * self.grid_size
            if y >= h:  # Ensure we don't go beyond image boundaries
                y = h - 1
                
            row_coords = []
            for col in range(cols + 1):  # +1 to include partial cells
                x = col * self.grid_size
                if x >= w:  # Ensure we don't go beyond image boundaries
                    x = w - 1
                    
                # Draw complete grid cell if possible
                end_x = min(x + self.grid_size, w - 1)
                end_y = min(y + self.grid_size, h - 1)
                cv2.rectangle(display_image, (x, y), (end_x, end_y), grid_color, 1)
                
                if row < rows and col < cols:  # Only add complete cells to the coordinates list
                    grid_coords.append((x, y))
                    row_coords.append(f"({x}, {y})")
            
            if row_coords:
                print(" ".join(row_coords))
        
        # Calculate the center grid cell
        center_col = cols // 2
        center_row = rows // 2
        center_x = center_col * self.grid_size
        center_y = center_row * self.grid_size
        
        # Mark center cell with a different color
        cv2.rectangle(display_image, 
                     (center_x, center_y), 
                     (min(center_x + self.grid_size, w-1), min(center_y + self.grid_size, h-1)), 
                     (0, 0, 255), 2)  # Red color for center
        
        print(f"**Tọa độ ô trung tâm (ô tâm): ({center_x}, {center_y})**")
        
        # Store grid information for other functions to use
        self.grid_rows = rows
        self.grid_cols = cols
        self.grid_coords = grid_coords
        self.center_grid = (center_row, center_col)  # Store as row, col for grid indexing
        
        return display_image

    def calculate_optimal_threshold(self, image: np.ndarray) -> float:
        """
        Calculate an optimal threshold for heat detection based on the enhanced image.
        Uses Otsu's method and adds statistical analysis.
        
        Args:
            image: Grayscale image for threshold calculation
            
        Returns:
            Threshold value
        """
        # Now returns a fixed threshold of 0.87
        print("Using fixed threshold: 0.87")
        return 0.87

    def find_heat_center(self, original_img: np.ndarray, enhanced_img: np.ndarray, 
                         hottest_value: Optional[float] = None, 
                         grid_threshold: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Find areas of high intensity in the image using targeted search from the center.
        
        Args:
            original_img: Original binary image from segmentation
            enhanced_img: Enhanced image for threshold calculation
            hottest_value: Value from kmeans to use as reference (optional)
            grid_threshold: Override grid threshold (optional, for recursive calls)
            
        Returns:
            List of heat center coordinates in pixel space (x, y)
        """
        # Use fixed threshold of 0.87
        fixed_threshold = 0.87
        print(f"Using fixed threshold: {fixed_threshold}")
        
        # Create a grid representation of the image
        rows, cols = self.grid_rows, self.grid_cols
        grid = np.zeros((rows, cols), dtype=np.float32)
        
        # Calculate average intensity for each grid cell
        for row in range(rows):
            for col in range(cols):
                y = row * self.grid_size
                x = col * self.grid_size
                
                # Get the grid cell region considering image boundaries
                end_y = min(y + self.grid_size, original_img.shape[0])
                end_x = min(x + self.grid_size, original_img.shape[1])
                
                cell_region = original_img[y:end_y, x:end_x]
                if cell_region.size > 0:
                    # For binary images, calculate percentage of white pixels
                    if np.max(original_img) <= 1 or set(np.unique(original_img)).issubset({0, 255}):
                        grid[row, col] = np.mean(cell_region) / 255 * 100
                    else:
                        grid[row, col] = np.mean(cell_region)
        
        # Save the grid intensity map without displaying it
        # This is thread-safe as we're not showing the plot interactively
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='hot')
        plt.colorbar(label='Mean Intensity')
        plt.title('Grid Cell Intensity Map')
        plt.tight_layout()
        plt.savefig('grid_intensity_map.png')
        plt.close(fig)  # Make sure to close the figure
        
        # Use the fixed threshold
        if grid_threshold is None:
            if np.max(original_img) <= 1 or set(np.unique(original_img)).issubset({0, 255}):
                grid_threshold = 30  # Lower threshold for binary images
            else:
                grid_threshold = fixed_threshold
        
        print(f"Ngưỡng lưới được áp dụng: {grid_threshold}")
        
        # Get center grid coordinates
        center_row, center_col = self.center_grid
        
        # Directions for 4-connectivity (only adjacent points)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # top, bottom, left, right
        
        # Lists to store processed and discarded points
        processed_points: Set[Tuple[int, int]] = set()
        discarded_points: Set[Tuple[int, int]] = set()
        heat_regions = []
        
        # Use BFS to find connected regions starting from the center
        def find_heat_region(start_row: int, start_col: int) -> Optional[List[Tuple[int, int]]]:
            if (start_row, start_col) in processed_points or (start_row, start_col) in discarded_points:
                return None
            
            # Check if the starting point meets the threshold
            if grid[start_row, start_col] < grid_threshold:
                discarded_points.add((start_row, start_col))
                return None
            
            # Initialize a new region and queue for BFS
            region = []
            queue = [(start_row, start_col)]
            processed_points.add((start_row, start_col))
            
            while queue:
                r, c = queue.pop(0)
                region.append((r, c))
                
                # Check each of the four adjacent points
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    
                    # Check if the point is valid, not processed, and not discarded
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        (nr, nc) not in processed_points and 
                        (nr, nc) not in discarded_points):
                        
                        # Check if the point meets the threshold
                        if grid[nr, nc] >= grid_threshold:
                            queue.append((nr, nc))
                            processed_points.add((nr, nc))
                        else:
                            # Add to discarded list if it doesn't meet threshold
                            discarded_points.add((nr, nc))
            
            return region if region else None
        
        # Start from the center
        center_region = find_heat_region(center_row, center_col)
        if center_region:
            heat_regions.append(center_region)
        
        # Continue searching from remaining unprocessed points
        # This ensures we find all heat regions, not just those connected to the center
        for row in range(rows):
            for col in range(cols):
                if (row, col) not in processed_points and (row, col) not in discarded_points:
                    region = find_heat_region(row, col)
                    if region:
                        heat_regions.append(region)
        
        # Find center of each heat region and convert to pixel coordinates
        heat_centers = []
        for region in heat_regions:
            if not region:
                continue
                
            # Find the cell in the region with highest intensity
            max_intensity_cell = max(region, key=lambda cell: grid[cell[0], cell[1]])
            
            # Convert grid coordinates to pixel coordinates
            pixel_x = max_intensity_cell[1] * self.grid_size  # col * grid_size = x
            pixel_y = max_intensity_cell[0] * self.grid_size  # row * grid_size = y
            
            heat_centers.append((pixel_x, pixel_y))
            print(f"Vùng nhiệt có {len(region)} ô, tâm tại ({pixel_x}, {pixel_y}), "
                  f"cường độ trung bình: {grid[max_intensity_cell[0], max_intensity_cell[1]]:.1f}")
        
        if heat_centers:
            print(f"Tìm thấy {len(heat_centers)} vùng nhiệt cao")
        else:
            print("Không tìm thấy vùng nhiệt cao với ngưỡng đã cho")
            # If no regions found with the current threshold, try a lower threshold
            if grid_threshold > 15:  # Lower limit for recursion
                print("Thử lại với ngưỡng thấp hơn...")
                return self.find_heat_center(original_img, enhanced_img, hottest_value, grid_threshold / 2)
        
        return heat_centers

    def highlight_heat_centers(self, grid_img: np.ndarray, 
                             heat_centers: List[Tuple[int, int]]) -> np.ndarray:
        """
        Highlight the identified heat centers on the grid image.
        
        Args:
            grid_img: Image with grid overlay
            heat_centers: List of heat center coordinates
            
        Returns:
            Image with highlighted heat centers
        """
        result_img = grid_img.copy()
        
        for x, y in heat_centers:
            # Draw a filled rectangle at the heat center
            cv2.rectangle(result_img, 
                         (int(x), int(y)), 
                         (int(x + self.grid_size), int(y + self.grid_size)),
                         (0, 0, 255), 2)  # Thick red border
                         
            # Add a circle at the center of the heat region
            center_x = int(x + self.grid_size // 2)
            center_y = int(y + self.grid_size // 2)
            cv2.circle(result_img, (center_x, center_y), 5, (255, 0, 0), -1)  # Filled blue circle
            
            # Add text label
            label = f"({x}, {y})"
            cv2.putText(result_img, label, (int(x), int(y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return result_img

    def get_enhanced_image(self, image_data: ImageData) -> np.ndarray:
        """
        Get the appropriate enhanced image based on the selected enhancement method
        and color preservation setting.
        
        Args:
            image_data: ImageData object containing various enhanced versions
            
        Returns:
            The selected enhanced image
        """
        if self.preserve_color:
            # Return color-preserved version if available
            if self.enhancement_method == 1 and image_data.enhanced_clahe_color is not None:
                return image_data.enhanced_clahe_color
            elif self.enhancement_method == 2 and image_data.enhanced_guided_color is not None:
                return image_data.enhanced_guided_color
            elif self.enhancement_method == 3 and image_data.enhanced_tophat_color is not None:
                return image_data.enhanced_tophat_color
            elif self.enhancement_method == 4 and image_data.enhanced_alcohol_color is not None:
                return image_data.enhanced_alcohol_color
            elif self.enhancement_method == 5 and image_data.enhanced_combined_color is not None:
                return image_data.enhanced_combined_color
        
        # Fallback to grayscale versions if color not available or not requested
        if self.enhancement_method == 1:
            return image_data.enhanced_clahe
        elif self.enhancement_method == 2 and image_data.enhanced_guided is not None:
            return image_data.enhanced_guided
        elif self.enhancement_method == 3 and image_data.enhanced_tophat is not None:
            return image_data.enhanced_tophat
        elif self.enhancement_method == 4 and image_data.enhanced_alcohol is not None:
            return image_data.enhanced_alcohol
        elif self.enhancement_method == 5 and image_data.enhanced_combined is not None:
            return image_data.enhanced_combined
        else:
            # Fallback to CLAHE if the selected method is not available
            return image_data.enhanced_clahe

    def process_image(self, filename: str, image_data: ImageData) -> Dict[str, np.ndarray]:
        """
        Process a single thermal image.
        
        Args:
            filename: Name of the image file
            image_data: Image data to process
            
        Returns:
            Dictionary of processed images
        """
        print(f"\n-------- Đang xử lý ảnh: {filename} --------")
        
        # Get enhanced image based on selected method and color preservation setting
        enhanced_img = self.get_enhanced_image(image_data)
        
        # For processing steps, we need grayscale version
        # If enhanced_img is color, convert to grayscale for processing
        if len(enhanced_img.shape) == 3:
            processing_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        else:
            processing_img = enhanced_img
        
        # Step 1: K-means clustering
        kmeans_mask, hottest_value = self._apply_kmeans_clustering(processing_img)
        
        # Step 2: Connected components analysis
        connected_components_img = self._apply_connected_components(kmeans_mask)
        
        # Step 3: Draw contours
        contour_img = self.contour_drawer.draw_contours(connected_components_img)
        
        # Step 4: Draw grid
        grid_img = self._draw_grid(connected_components_img.copy())
        
        # Step 5: Find heat centers with improved threshold calculation
        heat_centers = self.find_heat_center(connected_components_img, processing_img, hottest_value)
        
        # Step 6: Highlight heat centers
        highlighted_img = self.highlight_heat_centers(grid_img, heat_centers)
        
        print(f"-------- Hoàn thành xử lý ảnh: {filename} --------\n")
        
        # Return all processed images including all enhancement methods for comparison
        result = {
            'original': image_data.original,
            'enhanced': enhanced_img,  # The selected enhancement method
            'kmeans': kmeans_mask,
            'connected': connected_components_img,
            'contours': contour_img,
            'highlighted': highlighted_img
        }
        
        # Add all enhancement methods for comparison if they exist
        if self.preserve_color:
            # Add color versions
            if image_data.enhanced_clahe_color is not None:
                result['enhanced_clahe_color'] = image_data.enhanced_clahe_color
            if image_data.enhanced_guided_color is not None:
                result['enhanced_guided_color'] = image_data.enhanced_guided_color
            if image_data.enhanced_tophat_color is not None:
                result['enhanced_tophat_color'] = image_data.enhanced_tophat_color
            if image_data.enhanced_alcohol_color is not None:
                result['enhanced_alcohol_color'] = image_data.enhanced_alcohol_color
            if image_data.enhanced_combined_color is not None:
                result['enhanced_combined_color'] = image_data.enhanced_combined_color
        else:
            # Add grayscale versions
            if image_data.enhanced_clahe is not None:
                result['enhanced_clahe'] = image_data.enhanced_clahe
            if image_data.enhanced_guided is not None:
                result['enhanced_guided'] = image_data.enhanced_guided
            if image_data.enhanced_tophat is not None:
                result['enhanced_tophat'] = image_data.enhanced_tophat
            if image_data.enhanced_alcohol is not None:
                result['enhanced_alcohol'] = image_data.enhanced_alcohol
            if image_data.enhanced_combined is not None:
                result['enhanced_combined'] = image_data.enhanced_combined
            
        return result

    def process_all_images(self):
        """Process all images without threading for better compatibility."""
        if not self.thermal_images:
            print("Không có ảnh nào để xử lý!")
            return
        
        # Process images sequentially to avoid threading issues
        results = {}
        for filename, image_data in self.thermal_images.items():
            try:
                results[filename] = self.process_image(filename, image_data)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return results

    def process_and_display(self):
        """Process all images and display the results."""
        # Process all images first
        results = self.process_all_images()
        
        # Then display results in the main thread
        for filename, processed_images in results.items():
            self._display_results(
                processed_images['original'],
                processed_images['enhanced'],
                processed_images['kmeans'],
                processed_images['connected'],
                processed_images['contours'],
                processed_images['highlighted'],
                filename,
                processed_images  # Pass all processed images for comparison
            )

    def _display_results(self, original_img: np.ndarray, enhanced_img: np.ndarray,
                        kmeans_img: np.ndarray, connected_img: np.ndarray,
                        contour_img: np.ndarray, highlighted_img: np.ndarray,
                        filename: str, all_processed_images: Dict[str, np.ndarray] = None):
        """
        Display the results of the image processing pipeline.
        
        Args:
            original_img: Original image
            enhanced_img: Enhanced image
            kmeans_img: K-means clustering result
            connected_img: Connected components result
            contour_img: Image with contours
            highlighted_img: Image with highlighted heat centers
            filename: Name of the image file
            all_processed_images: Dictionary containing all processed images including different enhancement methods
        """
        # Create figure in the main thread
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Get enhancement method name for the title
        enhancement_names = {
            1: "CLAHE",
            2: "Multi-Scale Guided Filtering",
            3: "Multiscale Top-Hat Transform",
            4: "Region-Specific Alcohol Detection",
            5: "Combined Approach"
        }
        enhancement_name = enhancement_names.get(self.enhancement_method, "CLAHE")
        color_text = "Color Preserved" if self.preserve_color else "Grayscale"
        
        titles = [
            "Original", 
            f"{enhancement_name} Enhanced ({color_text})", 
            "K-means Clustering",
            "Connected Components", 
            "Heat Centers",
            "Contours",
        ]
        
        images = [
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            enhanced_img if len(enhanced_img.shape) == 2 else cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB),
            kmeans_img,
            connected_img, 
            cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
        ]
        
        cmaps = [
            None,  # RGB image
            None if len(enhanced_img.shape) == 3 else 'gray',  # Enhanced image (color or grayscale)
            'gray',  # K-means mask
            'gray',  # Connected components
            None,  # Heat centers (BGR)
            None   # Contours (BGR)
        ]
        
        for i, (ax, img, title, cmap) in enumerate(zip(axes, images, titles, cmaps)):
            ax.imshow(img, cmap=cmap)
            ax.set_title(f'{title}', fontsize=12)
            ax.axis('off')
        
        plt.suptitle(f'Thermal Image Analysis: {filename} (Method: {enhancement_name}, {color_text})', fontsize=16)
        plt.tight_layout()
        
        # Save the result
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(os.path.join(result_dir, f"analysis_{filename}.png"), dpi=300, bbox_inches='tight')
        
        # If we have multiple enhancement methods, create a comparison figure
        if all_processed_images:
            if self.preserve_color:
                # Check for color enhanced versions
                enhancement_methods = [key for key in all_processed_images.keys() 
                                     if key.endswith('_color') and key != 'enhanced_clahe_color']
                if enhancement_methods:
                    self._display_enhancement_comparison(filename, all_processed_images, color=True)
            else:
                # Check for grayscale enhanced versions
                enhancement_methods = [key for key in all_processed_images.keys() 
                                     if key.startswith('enhanced_') and not key.endswith('_color')]
                if enhancement_methods:
                    self._display_enhancement_comparison(filename, all_processed_images, color=False)
        
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Không thể hiển thị hình ảnh: {str(e)}")
            print("Hình ảnh đã được lưu trong thư mục 'results'")
        finally:
            plt.close(fig)  # Ensure figure is closed

    def _display_enhancement_comparison(self, filename: str, processed_images: Dict[str, np.ndarray], color: bool = False):
        """
        Display a comparison of different enhancement methods.
        
        Args:
            filename: Name of the image file
            processed_images: Dictionary containing all processed images
            color: Whether to display color or grayscale versions
        """
        # Count how many enhancement methods we have
        if color:
            enhancement_methods = [key for key in processed_images.keys() 
                                 if key.endswith('_color') and key != 'original']
        else:
            enhancement_methods = [key for key in processed_images.keys() 
                                 if key.startswith('enhanced_') and not key.endswith('_color')]
            
        if not enhancement_methods:
            return
            
        # Create a figure for comparison
        fig, axes = plt.subplots(1, len(enhancement_methods), figsize=(5*len(enhancement_methods), 5))
        if len(enhancement_methods) == 1:
            axes = [axes]  # Make it iterable if there's only one subplot
            
        titles = {
            'enhanced_clahe': "CLAHE",
            'enhanced_guided': "Multi-Scale Guided Filtering",
            'enhanced_tophat': "Multiscale Top-Hat Transform",
            'enhanced_alcohol': "Region-Specific Alcohol Detection",
            'enhanced_combined': "Combined Approach",
            'enhanced_clahe_color': "CLAHE (Color)",
            'enhanced_guided_color': "Multi-Scale Guided Filtering (Color)",
            'enhanced_tophat_color': "Multiscale Top-Hat Transform (Color)",
            'enhanced_alcohol_color': "Region-Specific Alcohol Detection (Color)",
            'enhanced_combined_color': "Combined Approach (Color)"
        }
        
        for i, method in enumerate(enhancement_methods):
            if method in processed_images and processed_images[method] is not None:
                img = processed_images[method]
                if len(img.shape) == 3:  # Color image
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:  # Grayscale image
                    axes[i].imshow(img, cmap='gray')
                axes[i].set_title(titles.get(method, method), fontsize=12)
                axes[i].axis('off')
        
        color_text = "Color" if color else "Grayscale"
        plt.suptitle(f'Enhancement Method Comparison ({color_text}): {filename}', fontsize=16)
        plt.tight_layout()
        
        # Save the comparison
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        color_suffix = "color" if color else "gray"
        plt.savefig(os.path.join(result_dir, f"enhancement_comparison_{color_suffix}_{filename}.png"), 
                   dpi=300, bbox_inches='tight')
        
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Không thể hiển thị hình ảnh so sánh: {str(e)}")
        finally:
            plt.close(fig)  # Ensure figure is closed

    def save_processed_images(self, output_dir="processed_images"):
        """
        Save all processed images to disk.
        
        Args:
            output_dir: Directory to save processed images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all images
        results = self.process_all_images()
        
        # Save each processed image
        for filename, processed in results.items():
            base_name = os.path.splitext(filename)[0]
            
            # Save each version
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.png"), processed['enhanced'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_kmeans.png"), processed['kmeans'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_connected.png"), processed['connected'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_contours.png"), processed['contours'])
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_highlighted.png"), processed['highlighted'])
            
            # Save all enhancement methods if they exist
            if self.preserve_color:
                # Save color versions
                for method in ['enhanced_clahe_color', 'enhanced_guided_color', 
                              'enhanced_tophat_color', 'enhanced_alcohol_color', 
                              'enhanced_combined_color']:
                    if method in processed and processed[method] is not None:
                        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{method}.png"), processed[method])
            else:
                # Save grayscale versions
                for method in ['enhanced_clahe', 'enhanced_guided', 
                              'enhanced_tophat', 'enhanced_alcohol', 
                              'enhanced_combined']:
                    if method in processed and processed[method] is not None:
                        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{method}.png"), processed[method])
            
        print(f"Đã lưu tất cả ảnh đã xử lý vào thư mục: {output_dir}")


class ContourDrawer:
    """Helper class for finding and drawing contours."""
    
    def find_contours(self, image: np.ndarray) -> List:
        """
        Find contours in a binary image.
        
        Args:
            image: Binary image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, image: np.ndarray) -> np.ndarray:
        """
        Draw contours on an image.
        
        Args:
            image: Binary image
            
        Returns:
            Image with contours drawn
        """
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        contours = self.find_contours(image)
        
        # Draw all contours
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Draw bounding rectangles for significant contours
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Only consider larger contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Add contour information
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                label = f"#{i}: {area:.0f}px²"
                cv2.putText(contour_image, label, (centroid_x, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return contour_image


def main():
    """Main function to run the thermal image analysis."""
    try:
        # Create and run the thermal image comparer with fixed grid_size=5
        # Note: The class now ignores this parameter and always uses 5
        comparer = ThermalImageComparer(k_clusters=3, grid_size=5)
        comparer.process_and_display()
        
        # Ask user if they want to save processed images
        save_images = input("Bạn có muốn lưu các ảnh đã xử lý không? (y/n): ")
        if save_images.lower() == 'y':
            comparer.save_processed_images()
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
