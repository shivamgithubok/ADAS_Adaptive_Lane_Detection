import math
import numpy as np
from adas.config.settings import config

class Validator:
    @staticmethod
    def validate(obj_data, seg_data, contact_data, track_info):
        """
        Validates the extracted vehicle data with a quality score.
        Returns: (score: int, status: str, reason: str)
        """
        # Hard rejections
        if seg_data is None or 'area' not in seg_data or seg_data['area'] <= 0:
            return 0, "FAIL", "Mask missing"
            
        if contact_data is None:
            return 0, "FAIL", "Ground strip empty"
            
        if contact_data.get('ground_pixels', 0) == 0:
            return 0, "FAIL", "No ground pixels"
            
        if contact_data.get('ground_width', 0) <= 0:
            return 0, "FAIL", "Ground width <= 0"
            
        left_pt = contact_data.get('left_contact', (np.nan, np.nan))
        right_pt = contact_data.get('right_contact', (np.nan, np.nan))
        
        if math.isnan(left_pt[0]) or math.isnan(right_pt[0]):
            return 0, "FAIL", "NaN coordinates"
            
        if left_pt == right_pt:
            return 0, "FAIL", "Less than two contact points"

        # Quality Score Calculation
        weights = config.QUALITY_THRESHOLDS
        
        # Mask Area component
        area_score = min(1.0, seg_data['area'] / 500.0)
        
        # Ground Pixels component
        gp_score = min(1.0, contact_data['ground_pixels'] / 20.0)
        
        # Ground Width component
        gw_score = min(1.0, contact_data['ground_width'] / 15.0)
        
        # Contact Geometry component (are points logically placed?)
        geom_score = 1.0 if left_pt[0] < right_pt[0] else 0.5
        
        total_score = (
            area_score * weights['mask_area_weight'] +
            gp_score * weights['ground_pixels_weight'] +
            gw_score * weights['ground_width_weight'] +
            geom_score * weights['contact_geometry_weight']
        )
        
        final_score = int(min(100, max(0, total_score)))
        
        if final_score >= 80:
            status = "PASS"
            reason = ""
        elif final_score >= 60:
            status = "LOW_CONFIDENCE"
            reason = "Score between 60-79"
        else:
            status = "FAIL"
            reason = "Score below 60"
            
        return final_score, status, reason
