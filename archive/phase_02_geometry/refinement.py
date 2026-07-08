import numpy as np
import math
import cv2
from dataclasses import dataclass

@dataclass
class RefinedFootprintGeometry:
    vehicle_id: int
    class_name: str
    observed_width_px: float
    observed_length_px: float
    refined_width_px: float
    refined_length_px: float
    refined_center: tuple
    refined_polygon: np.ndarray
    refined_rect: tuple
    projection_confidence: float
    geometry_confidence: float
    final_confidence: float
    correction_pct_w: float
    correction_pct_l: float
    distance_m: float
    history_frames: int

class FootprintRefiner:
    def __init__(self, homography):
        self.homography = homography
        # Class priors: (width_m, length_m)
        self.priors = {
            "car": (1.9, 4.5),
            "suv": (2.1, 4.8),
            "truck": (2.5, 6.5),
            "bus": (2.55, 12.0),
            "motorcycle": (0.8, 2.2),
            "default": (2.0, 4.5)
        }
        self.history = {}

    def process(self, fg, class_name: str, debug_data: dict) -> RefinedFootprintGeometry:
        # Convert px to m
        MPP = self.homography.MPP
        PPM = self.homography.PPM
        
        obs_w_m = fg.rect_width * MPP
        obs_l_m = fg.rect_length * MPP
        
        cls = class_name.lower()
        prior_w, prior_l = self.priors.get(cls, self.priors["default"])
        
        # Calculate distance (Z) from center
        # Center in pixel space (BEV canvas) -> Origin is at (400, 800)
        # y_px = 800 - Z_m * PPM => Z_m = (800 - y_px) / PPM
        z_m = (800.0 - fg.center_point[1]) / PPM
        if z_m < 0: z_m = 0
        
        # Distance weight: Trust obs more when near, trust prior more when far
        # Exponential decay: w_obs = e^(-k * Z)
        k = 0.05
        w_obs = math.exp(-k * z_m)
        w_obs = max(0.1, min(0.9, w_obs))
        
        # Refine Width (Blend)
        refined_w_m = w_obs * obs_w_m + (1.0 - w_obs) * prior_w
        
        # Refine Length
        # The observed length is basically just the rear bumper and a bit of side. 
        # So we almost entirely trust the prior length, but allow slight modification
        refined_l_m = (w_obs * 0.3) * obs_l_m + (1.0 - w_obs * 0.3) * prior_l
        
        # EMA Smoothing
        alpha = 0.3
        hist_len = 1
        if fg.vehicle_id in self.history:
            hist = self.history[fg.vehicle_id]
            hist_len = hist['frames'] + 1
            refined_w_m = alpha * refined_w_m + (1 - alpha) * hist['w']
            refined_l_m = alpha * refined_l_m + (1 - alpha) * hist['l']
            self.history[fg.vehicle_id] = {'w': refined_w_m, 'l': refined_l_m, 'frames': hist_len}
        else:
            self.history[fg.vehicle_id] = {'w': refined_w_m, 'l': refined_l_m, 'frames': 1}
            
        # Reconstruct full footprint
        # We assume observed center is near the rear bumper.
        # We need to shift the center forward along the PCA heading by (refined_l_m / 2)
        # PCA vector gives direction. We want the direction pointing "forward" into the image 
        # In BEV, Y=0 is top, Y=800 is bottom (ego). So "forward" means negative Y.
        heading = np.array(fg.orientation_vector)
        if heading[1] > 0: 
            heading = -heading
            
        # Shift true center forward by shift_m
        shift_m = refined_l_m / 2.0
        shift_px = shift_m * PPM
        ref_center_px = (
            fg.center_point[0] + heading[0] * shift_px,
            fg.center_point[1] + heading[1] * shift_px
        )
        
        # Rebuild polygon
        ref_w_px = refined_w_m * PPM
        ref_l_px = refined_l_m * PPM
        
        ortho = np.array([-heading[1], heading[0]])
        
        hl = ref_l_px / 2.0
        hw = ref_w_px / 2.0
        
        p1 = (ref_center_px[0] + heading[0]*hl + ortho[0]*hw, ref_center_px[1] + heading[1]*hl + ortho[1]*hw) # FL
        p2 = (ref_center_px[0] + heading[0]*hl - ortho[0]*hw, ref_center_px[1] + heading[1]*hl - ortho[1]*hw) # FR
        p3 = (ref_center_px[0] - heading[0]*hl - ortho[0]*hw, ref_center_px[1] - heading[1]*hl - ortho[1]*hw) # RR
        p4 = (ref_center_px[0] - heading[0]*hl + ortho[0]*hw, ref_center_px[1] - heading[1]*hl + ortho[1]*hw) # RL
        
        refined_poly = np.array([p1, p2, p3, p4], dtype=np.float32)
        refined_rect = cv2.minAreaRect(refined_poly)
        
        # Confidence Calculation
        proj_conf = debug_data.get("ground_density", 50.0) * 0.5 + debug_data.get("mask_completeness", 50.0) * 0.5
        geom_conf = debug_data.get("hull_consistency", 50.0)
        
        base_conf = fg.quality_score
        dist_penalty = min(30.0, z_m * 0.5)
        hist_reward = min(20.0, hist_len * 2.0)
        final_conf = max(0.0, min(100.0, base_conf - dist_penalty + hist_reward))
        
        corr_w = (ref_w_px - fg.rect_width) / max(1.0, fg.rect_width) * 100.0
        corr_l = (ref_l_px - fg.rect_length) / max(1.0, fg.rect_length) * 100.0

        rfg = RefinedFootprintGeometry(
            vehicle_id=fg.vehicle_id,
            class_name=class_name,
            observed_width_px=fg.rect_width,
            observed_length_px=fg.rect_length,
            refined_width_px=ref_w_px,
            refined_length_px=ref_l_px,
            refined_center=ref_center_px,
            refined_polygon=refined_poly,
            refined_rect=refined_rect,
            projection_confidence=proj_conf,
            geometry_confidence=geom_conf,
            final_confidence=final_conf,
            correction_pct_w=corr_w,
            correction_pct_l=corr_l,
            distance_m=z_m,
            history_frames=hist_len
        )
        return rfg
