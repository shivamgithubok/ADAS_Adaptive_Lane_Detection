from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any, Dict
import numpy as np

@dataclass
class Configuration:
    """Base Configuration class."""
    pass

@dataclass
class GroundContact:
    left_contact: Tuple[float, float]
    right_contact: Tuple[float, float]
    median_contact: Tuple[float, float]
    ground_width: float
    mask_height: int
    strip_height: int
    ground_pixels: int

@dataclass
class BEVPoint:
    x: float
    y: float

@dataclass
class BEVFootprint:
    fl: BEVPoint
    fr: BEVPoint
    rl: BEVPoint
    rr: BEVPoint
    polygon: List[BEVPoint]

@dataclass
class VehicleState:
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    segmentation_mask: np.ndarray
    ground_contact: Optional[GroundContact] = None
    track_age: int = 0
    timestamp: float = 0.0
    status: str = "UNKNOWN"
    score: float = 0.0

@dataclass
class VehicleGeometry:
    id: int
    class_name: str
    confidence: float
    img_left: Tuple[int, int]
    img_right: Tuple[int, int]
    img_center: Tuple[int, int]
    bev_left: Tuple[float, float]
    bev_right: Tuple[float, float]
    bev_center: Tuple[float, float]
    width_px: int
    width_m: float
    length_m: float
    footprint_fl: Tuple[float, float]
    footprint_fr: Tuple[float, float]
    footprint_rl: Tuple[float, float]
    footprint_rr: Tuple[float, float]
    footprint: Any = None
    metric_estimate: Any = None
    dim_estimate: Any = None

@dataclass
class LanePolyline:
    points: List[Tuple[int, int]]
    bev_points: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    is_solid: bool = False

@dataclass
class LaneDetection:
    left_lane: Optional[LanePolyline]
    right_lane: Optional[LanePolyline]
    center_lane: Optional[LanePolyline] = None
    lane_width_m: float = 0.0
    curvature: float = 0.0

@dataclass
class TrackerState:
    active_tracks: int
    history: Dict[int, Any]

@dataclass
class FrameData:
    frame_count: int
    timestamp: float
    image: np.ndarray
    vehicles: List[VehicleState] = field(default_factory=list)
    lane_detection: Optional[LaneDetection] = None
    geometry_objects: List[VehicleGeometry] = field(default_factory=list)
