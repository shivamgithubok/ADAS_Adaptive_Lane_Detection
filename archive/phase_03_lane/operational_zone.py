from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ODZConfig:
    MAX_FORWARD_DISTANCE: float = 40.0
    MAX_LATERAL_DISTANCE: float = 5.55
    MIN_MASK_HEIGHT: int = 20
    MIN_GROUND_PIXELS: int = 10
    MIN_GROUND_WIDTH: int = 10
    MIN_VISIBILITY_SCORE: float = 50.0
    IMAGE_BORDER_MARGIN: int = 5

@dataclass
class VehicleODZState:
    vehicle_id: int
    status: str # "ACTIVE" or "INACTIVE"
    reason: str
    geometry: Any # VehicleGeometry reference
    bbox: List[int] # Original bounding box
    planning_eligible: bool = False
    occlusion_state: Optional[Any] = None  # VehicleOcclusionState, if available

class ODZFilter:
    def __init__(self, config: Optional[ODZConfig] = None):
        self.config = config or ODZConfig()
        
    def process(self, geometry_objects: List[Any], detected_objects: List[Dict[str, Any]], frame_w: int, frame_h: int, homography: Any = None) -> List[VehicleODZState]:
        states = []
        
        # Map detected objects by track_id for easy lookup
        det_map = {obj['track_id']: obj for obj in detected_objects}
        
        PPM = homography.PPM if homography else 145.0 / 3.7
        
        for vg in geometry_objects:
            vme = vg.metric_estimate
            fg = vg.footprint
            vehicle_id = vg.id
            
            obj = det_map.get(vehicle_id)
            if not obj:
                states.append(VehicleODZState(vehicle_id, "INACTIVE", "No Detection Data", vg, [0,0,0,0], False))
                continue
                
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            bbox_height = y2 - y1
            
            status = "ACTIVE"
            reason = ""
            
            # 1. Image Border Filter (top, left, right)
            margin = self.config.IMAGE_BORDER_MARGIN
            if x1 <= margin or x2 >= frame_w - margin or y1 <= margin:
                status = "INACTIVE"
                reason = "Image Border"
                
            # 2. Projection Check
            elif not vme or not fg:
                status = "INACTIVE"
                reason = "Projection Failure"
                
            # 3. Behind Ego Check
            elif fg.center_point[1] > 800:
                status = "INACTIVE"
                reason = "Behind Ego"
                
            # 4. Distance Filter
            elif vme and vme.distance_m > self.config.MAX_FORWARD_DISTANCE:
                status = "INACTIVE"
                reason = "Far Distance"
                
            # 5. Lateral Distance Filter
            elif abs((fg.center_point[0] - 400) / PPM) > self.config.MAX_LATERAL_DISTANCE:
                status = "INACTIVE"
                reason = "Outside Operational Zone"
                
            # 6. Minimum Size Filter
            elif bbox_height < self.config.MIN_MASK_HEIGHT:
                status = "INACTIVE"
                reason = "Tiny Vehicle"
                
            # 7. Visibility Filter
            elif vme and vme.correction_confidence < self.config.MIN_VISIBILITY_SCORE:
                status = "INACTIVE"
                reason = "Low Confidence"
                
            planning = (status == "ACTIVE")
            states.append(VehicleODZState(vehicle_id, status, reason, vg, bbox, planning))
            
        return states
