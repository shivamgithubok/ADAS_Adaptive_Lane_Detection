import cv2
import numpy as np
from pathlib import Path
import logging
from adas.core.FreeSpace.utils import apply_color_map

logger = logging.getLogger(__name__)

class FreeSpaceVisualizer:
    """
    Handles drawing the free-space overlay and depth visualizations
    and saving the results.
    """
    def __init__(self, output_dir: str = 'outputs', alpha: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha  # Transparency for overlay
        
        # Overlay color (B, G, R) - Semi-transparent green
        self.overlay_color = (0, 255, 0)
        
    def draw_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Overlays the binary free-space mask on the RGB image.
        
        Args:
            image (np.ndarray): Original RGB image (HxWxC).
            mask (np.ndarray): Binary mask (HxW).
            
        Returns:
            np.ndarray: Blended image.
        """
        # Create a solid color image for the overlay
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:] = self.overlay_color
        
        # Use mask to extract the overlay region
        overlay = cv2.bitwise_and(color_layer, color_layer, mask=mask)
        
        # Prepare the base image: keep original pixels where mask is 0
        inv_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(image, image, mask=inv_mask)
        
        # Blend the original image and the colored layer in the masked region
        blended_region = cv2.addWeighted(image, 1 - self.alpha, overlay, self.alpha, 0)
        blended_region = cv2.bitwise_and(blended_region, blended_region, mask=mask)
        
        # Combine
        final_overlay = cv2.add(background, blended_region)
        return final_overlay
        
    def save_visualization(self, original_image: np.ndarray, freespace_result: dict, filename: str) -> None:
        """
        Creates a combined visualization (Original -> Overlay -> Depth) and saves it.
        
        Args:
            original_image: The raw RGB input.
            freespace_result: The output from FreeSpaceDetector.
            filename: Name to save the file as.
        """
        depth_map = freespace_result['depth']
        mask = freespace_result['mask']
        
        # 1. Generate Overlay
        overlay_img = self.draw_overlay(original_image, mask)
        
        # 2. Colorize Depth Map
        colored_depth = apply_color_map(depth_map)
        
        # 3. Concatenate horizontally
        # Add thin separator lines
        h = original_image.shape[0]
        separator = np.ones((h, 10, 3), dtype=np.uint8) * 255
        
        combined = cv2.hconcat([
            original_image, 
            separator, 
            overlay_img, 
            separator, 
            colored_depth
        ])
        
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), combined)
        logger.info(f"Saved visualization to {save_path}")
        
        return combined
