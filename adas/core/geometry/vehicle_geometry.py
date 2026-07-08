from dataclasses import dataclass
from typing import Tuple

@dataclass
class VehicleGeometry:
    id: int
    class_name: str
    confidence: float
    
    # Image points
    img_left: Tuple[int, int]
    img_right: Tuple[int, int]
    img_center: Tuple[int, int]
    
    # BEV coordinates (meters)
    bev_left: Tuple[float, float]
    bev_right: Tuple[float, float]
    bev_center: Tuple[float, float]
    
    # Measured dimensions
    width_px: int
    width_m: float
    length_m: float
    
    # Axis-aligned footprint corners in BEV (meters)
    # Order: Front-Left, Front-Right, Rear-Left, Rear-Right
    footprint_fl: Tuple[float, float]
    footprint_fr: Tuple[float, float]
    footprint_rl: Tuple[float, float]
    footprint_rr: Tuple[float, float]
