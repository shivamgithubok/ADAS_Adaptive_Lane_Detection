import cv2
import numpy as np
import config

class ContactRegionExtractor:
    @staticmethod
    def extract(mask_img, bbox):
        """
        Extracts the road-contact region of the vehicle from the mask.
        mask_img: Binary mask of the vehicle (numpy array same size as frame)
        bbox: (x1, y1, x2, y2) - not used for strip calculation anymore
        """
        # Get all mask pixels
        y_all, x_all = np.where(mask_img > 0)
        
        if len(y_all) == 0:
            return None
            
        y_min = np.min(y_all)
        y_max = np.max(y_all)
        x_min = np.min(x_all)
        x_max = np.max(x_all)
        
        mask_height = y_max - y_min + 1
        mask_width = x_max - x_min + 1
        
        # 1. Determine strip height using clamping
        raw_strip = int(mask_height * config.STRIP_RATIO)
        strip_height = max(config.MIN_STRIP_PIXELS, min(config.MAX_STRIP_PIXELS, raw_strip))
        
        # Determine strip boundaries
        strip_y1 = max(0, y_max - strip_height)
        strip_y2 = y_max + 1
        strip_x1 = x_min
        strip_x2 = x_max + 1
        
        # Create strip mask
        strip_mask = np.zeros_like(mask_img)
        strip_mask[strip_y1:strip_y2, strip_x1:strip_x2] = mask_img[strip_y1:strip_y2, strip_x1:strip_x2]
        
        # 3. Find mask pixels inside the strip
        y_coords, x_coords = np.where(strip_mask > 0)
        
        if len(x_coords) == 0:
            return {
                "mask_height": mask_height,
                "mask_width": mask_width,
                "strip_height": strip_height,
                "ground_pixels": 0,
                "left_contact": (np.nan, np.nan),
                "right_contact": (np.nan, np.nan),
                "median_contact": (np.nan, np.nan),
                "ground_width": 0,
                "strip_bbox": (strip_x1, strip_y1, strip_x2, strip_y2)
            }
            
        # 4. Compute boundaries
        left_x = np.min(x_coords)
        right_x = np.max(x_coords)
        
        # Find the y coordinate for the extreme x points to return an exact pixel
        left_y_candidates = y_coords[x_coords == left_x]
        right_y_candidates = y_coords[x_coords == right_x]
        
        # Pick the lowest (max y) for contact
        left_y = np.max(left_y_candidates)
        right_y = np.max(right_y_candidates)
        
        left_contact = (int(left_x), int(left_y))
        right_contact = (int(right_x), int(right_y))
        
        median_x = int(np.median(x_coords))
        median_y_candidates = y_coords[x_coords == median_x]
        median_y = int(np.max(median_y_candidates)) if len(median_y_candidates) > 0 else int(np.max(y_coords))
        median_contact = (median_x, median_y)
        
        road_contact_width = right_x - left_x
        
        ground_pixels = len(x_coords)
            
        return {
            "mask_height": mask_height,
            "mask_width": mask_width,
            "strip_height": strip_height,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "median_contact": median_contact,
            "ground_width": int(road_contact_width),
            "ground_pixels": ground_pixels,
            "strip_bbox": (strip_x1, strip_y1, strip_x2, strip_y2)
        }
