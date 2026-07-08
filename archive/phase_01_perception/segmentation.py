import cv2
import numpy as np

class SegmentationAnalyzer:
    @staticmethod
    def process_mask(mask_img):
        """
        Processes a binary mask image (numpy array of 0s and 255s)
        Returns:
            area: Mask Area (pixel count from countNonZero)
            contour: External contour points
            convex_hull: Convex hull of the contour
            bounding_rect: (x, y, w, h)
            pixel_count: Number of non-zero pixels
        """
        # Pixel count
        pixel_count = cv2.countNonZero(mask_img)
        area = pixel_count  # conceptually similar here
        
        # Find contour
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "area": pixel_count,
                "contour": [],
                "convex_hull": [],
                "bounding_rect": (0,0,0,0),
                "pixel_count": pixel_count
            }
            
        # Get the largest contour just in case
        main_contour = max(contours, key=cv2.contourArea)
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        
        # Bounding rect
        x, y, w, h = cv2.boundingRect(main_contour)
        
        return {
            "area": pixel_count,
            "contour": main_contour.reshape(-1, 2).tolist(),
            "convex_hull": hull.reshape(-1, 2).tolist(),
            "bounding_rect": (x, y, w, h),
            "pixel_count": pixel_count
        }
