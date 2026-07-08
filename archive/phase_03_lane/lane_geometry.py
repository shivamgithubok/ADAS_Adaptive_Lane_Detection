import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class LaneDetectionResult:
    left_polynomial: Optional[np.ndarray]  # [m, b] or [A, B, C]
    right_polynomial: Optional[np.ndarray]
    left_lane_pixels: int
    right_lane_pixels: int
    sampled_image_points_left: Optional[np.ndarray]
    sampled_image_points_right: Optional[np.ndarray]
    confidence: float
    visibility: float
    lane_type: str
    pixel_count: int

@dataclass
class LaneCurve:
    id: int
    image_polynomial: Optional[np.ndarray]
    bev_polynomial: Optional[np.ndarray]  # [A, B, C] for x = Ay^2 + By + C (in BEV space)
    poly_coeffs_m: Optional[np.ndarray] # [A, B, C] in metric space
    image_points: Optional[np.ndarray]
    bev_points: Optional[np.ndarray]
    sampled_points: Optional[np.ndarray]  # Dense BEV points for easy plotting
    curvature_m: float
    heading: float
    visibility: float
    confidence: float
    model_type: str

@dataclass
class RoadGeometry:
    left_lane: Optional[LaneCurve]
    right_lane: Optional[LaneCurve]
    centerline: Optional[LaneCurve]
    lane_width: float
