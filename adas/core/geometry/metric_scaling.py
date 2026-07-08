import numpy as np
import math
import cv2
from dataclasses import dataclass
from typing import Optional

@dataclass
class VehicleMetricEstimate:
    vehicle_id: int
    distance_m: float
    observed_width_px: float
    observed_length_px: float
    meters_per_pixel: float
    observed_width_m: float
    observed_length_m: float
    corrected_width_m: float
    corrected_length_m: float
    correction_confidence: float
    corrected_polygon: np.ndarray

class MetricScaler:
    def __init__(self, homography):
        self.homography = homography
        
    def process(self, fg, debug_data: dict) -> VehicleMetricEstimate:
        PPM = self.homography.PPM
        base_MPP = self.homography.MPP
        
        # Step 1: Distance and Local MPP
        z_m = (800.0 - fg.center_point[1]) / PPM
        if z_m < 0: z_m = 0.0
        
        # Smooth scaling function for perspective foreshortening
        # For close objects (< 10m), local_mpp is just base_mpp.
        # For far objects, it linearly increases to compensate for compression.
        k = 0.015  # Scale factor slope
        local_mpp = base_MPP
        if z_m > 10.0:
            local_mpp = base_MPP * (1.0 + k * (z_m - 10.0))
            
        # Step 2: Convert to observed meters using local MPP
        obs_w_m = fg.rect_width * local_mpp
        obs_l_m = fg.rect_length * local_mpp
        
        # Step 3 & 4: Estimate corrected dimensions with soft clamping
        # We don't want arbitrary constants like length *= 4.
        # We just soft-clamp to physically plausible bounds and blend based on distance.
        
        # Plausible bounds for generic vehicle
        min_w, max_w = 1.6, 2.2
        min_l, max_l = 3.5, 5.5
        
        # The further away, the more we trust the plausible bounds over the raw observation
        # w_prior goes from 0.0 at 10m to 0.8 at 60m
        w_prior = min(0.8, max(0.0, (z_m - 10.0) / 50.0))
        
        # Initial guess is just the observed value
        corr_w_m = obs_w_m
        corr_l_m = obs_l_m
        
        # Soft clamp Width
        if corr_w_m < min_w:
            corr_w_m = corr_w_m * (1 - w_prior) + min_w * w_prior
        elif corr_w_m > max_w:
            corr_w_m = corr_w_m * (1 - w_prior) + max_w * w_prior
            
        # Soft clamp Length
        if corr_l_m < min_l:
            corr_l_m = corr_l_m * (1 - w_prior) + min_l * w_prior
        elif corr_l_m > max_l:
            corr_l_m = corr_l_m * (1 - w_prior) + max_l * w_prior
            
        # Step 5: Correction Confidence
        base_conf = fg.quality_score
        dist_uncertainty = min(40.0, z_m * 0.5)
        # Using ground density as a proxy for footprint consistency
        consistency = debug_data.get("ground_density", 50.0)
        
        correction_conf = max(0.0, min(100.0, base_conf - dist_uncertainty + consistency * 0.2))
        
        # Rebuild polygon using the corrected dimensions
        # The center is unchanged. The heading is unchanged.
        heading = np.array(fg.orientation_vector)
        if heading[1] > 0: 
            heading = -heading
            
        ortho = np.array([-heading[1], heading[0]])
        
        # Convert corrected meters back to BEV pixels using base PPM for canvas drawing
        corr_w_px = corr_w_m * PPM
        corr_l_px = corr_l_m * PPM
        
        hl = corr_l_px / 2.0
        hw = corr_w_px / 2.0
        
        c_px = fg.center_point
        p1 = (c_px[0] + heading[0]*hl + ortho[0]*hw, c_px[1] + heading[1]*hl + ortho[1]*hw) # FL
        p2 = (c_px[0] + heading[0]*hl - ortho[0]*hw, c_px[1] + heading[1]*hl - ortho[1]*hw) # FR
        p3 = (c_px[0] - heading[0]*hl - ortho[0]*hw, c_px[1] - heading[1]*hl - ortho[1]*hw) # RR
        p4 = (c_px[0] - heading[0]*hl + ortho[0]*hw, c_px[1] - heading[1]*hl + ortho[1]*hw) # RL
        
        corrected_poly = np.array([p1, p2, p3, p4], dtype=np.float32)

        return VehicleMetricEstimate(
            vehicle_id=fg.vehicle_id,
            distance_m=z_m,
            observed_width_px=fg.rect_width,
            observed_length_px=fg.rect_length,
            meters_per_pixel=local_mpp,
            observed_width_m=obs_w_m,
            observed_length_m=obs_l_m,
            corrected_width_m=corr_w_m,
            corrected_length_m=corr_l_m,
            correction_confidence=correction_conf,
            corrected_polygon=corrected_poly
        )
