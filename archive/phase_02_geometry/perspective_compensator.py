import numpy as np
import math
import cv2
from dataclasses import dataclass
from typing import Optional
from refinement import RefinedFootprintGeometry

@dataclass
class CompensatedFootprint:
    vehicle_id: int
    corrected_width_m: float
    corrected_length_m: float
    corrected_width_px: float
    corrected_length_px: float
    correction_factor: float
    distance_m: float
    perspective_confidence: float
    corrected_polygon: np.ndarray
    original_polygon: np.ndarray
    heading: np.ndarray
    center_px: tuple

class PerspectiveCompensator:
    def __init__(self, homography):
        self.homography = homography

    def process(self, rfg: RefinedFootprintGeometry, heading_vector: tuple) -> CompensatedFootprint:
        z_m = rfg.distance_m
        
        # Calculate perspective correction factor gamma
        # If z <= 10m, gamma = 1.0
        # If z > 10m, gamma = 1.0 + k * (z - 10)
        gamma = 1.0
        if z_m > 10.0:
            kappa = 0.01 # 1% increase per meter beyond 10m
            gamma = 1.0 + kappa * (z_m - 10.0)
            
        # Clamp to a reasonable max scale to prevent explosions for very distant misdetections
        gamma = min(1.5, gamma)
        
        # Calculate compensated dimensions
        MPP = self.homography.MPP
        PPM = self.homography.PPM
        
        ref_w_m = rfg.refined_width_px * MPP
        ref_l_m = rfg.refined_length_px * MPP
        
        corrected_w_m = ref_w_m * gamma
        corrected_l_m = ref_l_m * gamma
        
        corrected_w_px = corrected_w_m * PPM
        corrected_l_px = corrected_l_m * PPM
        
        # Reconstruct the corrected polygon anchored symmetrically at the center
        heading = np.array(heading_vector)
        if heading[1] > 0:
            heading = -heading
            
        ortho = np.array([-heading[1], heading[0]])
        
        hl = corrected_l_px / 2.0
        hw = corrected_w_px / 2.0
        
        c_px = rfg.refined_center
        
        p1 = (c_px[0] + heading[0]*hl + ortho[0]*hw, c_px[1] + heading[1]*hl + ortho[1]*hw) # FL
        p2 = (c_px[0] + heading[0]*hl - ortho[0]*hw, c_px[1] + heading[1]*hl - ortho[1]*hw) # FR
        p3 = (c_px[0] - heading[0]*hl - ortho[0]*hw, c_px[1] - heading[1]*hl - ortho[1]*hw) # RR
        p4 = (c_px[0] - heading[0]*hl + ortho[0]*hw, c_px[1] - heading[1]*hl + ortho[1]*hw) # RL
        
        corrected_poly = np.array([p1, p2, p3, p4], dtype=np.float32)
        
        # Perspective Confidence
        # Higher distance -> lower perspective confidence due to more aggressive correction
        perspective_conf = max(0.0, min(100.0, rfg.final_confidence - (gamma - 1.0) * 100.0))
        
        return CompensatedFootprint(
            vehicle_id=rfg.vehicle_id,
            corrected_width_m=corrected_w_m,
            corrected_length_m=corrected_l_m,
            corrected_width_px=corrected_w_px,
            corrected_length_px=corrected_l_px,
            correction_factor=gamma,
            distance_m=z_m,
            perspective_confidence=perspective_conf,
            corrected_polygon=corrected_poly,
            original_polygon=rfg.refined_polygon,
            heading=heading,
            center_px=c_px
        )
