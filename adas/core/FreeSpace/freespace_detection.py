import cv2
import numpy as np
import logging

from .config import FreeSpaceConfig
from .depth_engine import DepthEngine

logger = logging.getLogger(__name__)

class FreeSpaceDetector:
    """
    Coordinates depth estimation and geometric reasoning to output
    a drivable free-space mask.
    """
    def __init__(self, config: FreeSpaceConfig = None):
        self.config = config if config else FreeSpaceConfig()
        self.engine = DepthEngine(self.config)
        
    def remove_sky(self, mask: np.ndarray) -> np.ndarray:
        """Removes the upper portion of the image, which is assumed to be sky."""
        height = mask.shape[0]
        sky_limit = int(height * self.config.sky_cutoff_ratio)
        result = mask.copy()
        result[:sky_limit, :] = 0
        return result

    def detect_obstacles(self, depth_map: np.ndarray) -> np.ndarray:
        """Detects objects based on local depth gradients (discontinuities)."""
        # Compute gradients in x and y
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # High gradients indicate depth discontinuities (obstacles)
        _, obstacle_mask = cv2.threshold(
            grad_mag, 
            self.config.depth_gradient_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        return obstacle_mask.astype(np.uint8)

    def estimate_ground(self, depth_map: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
        """
        Estimates the ground plane by assuming ground has smooth depth variations.
        Returns a mask of potential ground pixels.
        """
        # Potential ground is where there are no obstacles (smooth depth)
        ground_mask = cv2.bitwise_not(obstacle_mask)
        return ground_mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applies morphological operations (opening, closing, hole filling) to clean the mask."""
        ksize = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        
        # Opening: remove small noise points
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing: fill small holes and gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Hole filling using contours
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                # If a contour has a parent, it's a hole inside a larger region
                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(cleaned, [contour], 0, 255, -1)
                    
        return cleaned

    def get_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Finds the drivable region by extracting the largest connected component 
        that originates from the bottom center of the image.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return np.zeros_like(mask)
            
        height, width = mask.shape
        
        # Seed region: bottom center area, representing Ego vehicle's immediate path
        seed_h, seed_w = 50, 100
        y_start = max(0, height - seed_h)
        x_start = max(0, width // 2 - seed_w // 2)
        x_end = min(width, width // 2 + seed_w // 2)
        
        seed_region = labels[y_start:height, x_start:x_end]
        unique_labels, counts = np.unique(seed_region, return_counts=True)
        
        target_label = 0
        max_count = 0
        
        # Find the most prominent label in the seed region (excluding background 0)
        for label, count in zip(unique_labels, counts):
            if label == 0:
                continue
            if count > max_count:
                max_count = count
                target_label = label
                
        # Fallback: if no component touches the bottom center, just take the largest component
        if target_label == 0:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            target_label = largest_label
            
        result = np.zeros_like(mask)
        result[labels == target_label] = 255
        
        return result

    def extract_freespace(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Executes the geometric free-space extraction pipeline.
        """
        # Smooth depth map to reduce noise
        if self.config.median_blur_size > 0:
            smoothed_depth = cv2.medianBlur(depth_map, self.config.median_blur_size)
        else:
            smoothed_depth = depth_map
            
        # 1. Detect obstacles (depth discontinuities)
        obstacle_mask = self.detect_obstacles(smoothed_depth)
        
        # 2. Estimate ground plane (regions with smooth depth variation)
        ground_mask = self.estimate_ground(smoothed_depth, obstacle_mask)
        
        # 3. Remove Sky region
        ground_mask = self.remove_sky(ground_mask)
        
        # 4. Morphological Cleanup (Opening, Closing, Hole filling)
        cleaned_mask = self.clean_mask(ground_mask)
        
        # 5. Extract Largest Connected Component from bottom center seed
        freespace_mask = self.get_largest_connected_component(cleaned_mask)
        
        return freespace_mask

    def detect(self, image: np.ndarray) -> dict:
        """
        Processes an RGB image to detect free space.
        
        Args:
            image (np.ndarray): HxWxC RGB image.
            
        Returns:
            dict: {
                "depth": np.ndarray (HxW uint8 depth map),
                "mask": np.ndarray (HxW uint8 binary mask of free space),
                "polygon": None (placeholder for future)
            }
        """
        # 1. Infer depth
        depth_map = self.engine.infer(image)
        
        # 2. Geometric reasoning pipeline
        freespace_mask = self.extract_freespace(depth_map)
        
        return {
            "depth": depth_map,
            "mask": freespace_mask,
            "polygon": None
        }
