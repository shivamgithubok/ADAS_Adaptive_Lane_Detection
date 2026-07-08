import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from adas.core.projection.lane_projector import BEVRoadGeometry

@dataclass
class VehicleLaneAssignment:
    vehicle_id: int
    lane_id: str # "Center", "Left", "Right", "Unknown"
    lateral_offset_m: float # Offset from the centerline
    longitudinal_dist_m: float # Z distance
    dist_to_left_bound_m: float
    dist_to_right_bound_m: float
    overlaps_marking: bool

class LaneAssigner:
    def __init__(self, lane_width_m=3.7):
        self.lane_width_m = lane_width_m
        
    def _find_closest_point_on_curve(self, pt_m: Tuple[float, float], curve_pts_m: np.ndarray) -> Tuple[float, float, float]:
        """Finds closest point on a curve and returns (closest_x, closest_z, distance)"""
        if curve_pts_m is None or len(curve_pts_m) == 0:
            return 0.0, 0.0, float('inf')
            
        diffs = curve_pts_m - np.array(pt_m)
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)
        
        closest_pt = curve_pts_m[idx]
        return closest_pt[0], closest_pt[1], dists[idx]

    def assign(self, vehicles: List[any], bev_road: BEVRoadGeometry) -> List[VehicleLaneAssignment]:
        """
        vehicles: List of objects containing at least 'vehicle_id', and 'metric_estimate.distance_m' / center point
        We will extract center_x_m and center_z_m from the vehicle footprint.
        """
        assignments = []
        
        for v in vehicles:
            if not hasattr(v, 'metric_estimate') or v.metric_estimate is None:
                continue
                
            vme = v.metric_estimate
            
            # Vehicle center in meters (assuming camera is at x=0, z=0)
            # vme.distance_m is Z
            # To get X, we need the center point of the footprint in BEV
            # Or we can compute from vme.corrected_polygon
            if vme.corrected_polygon is not None and len(vme.corrected_polygon) > 0:
                # vme.corrected_polygon is in BEV px. We need to convert to meters.
                # Assuming PPM is available or we approximate:
                # However, vme itself has observed_width_m etc, but maybe we need absolute X_m.
                # Let's approximate from distance and heading, or if we have access to homography PPM:
                PPM = 145.0 / 3.70
                c_px_x = np.mean(vme.corrected_polygon[:, 0])
                c_x_m = (c_px_x - 400.0) / PPM
            else:
                c_x_m = 0.0
                
            c_z_m = vme.distance_m
            v_pt = (c_x_m, c_z_m)
            v_width = vme.corrected_width_m
            
            # Distances to curves
            dist_left = float('inf')
            dist_right = float('inf')
            dist_center = float('inf')
            
            if bev_road.left_lane and bev_road.left_lane.sampled_points_m is not None:
                _, _, dist_left = self._find_closest_point_on_curve(v_pt, bev_road.left_lane.sampled_points_m)
                
            if bev_road.right_lane and bev_road.right_lane.sampled_points_m is not None:
                _, _, dist_right = self._find_closest_point_on_curve(v_pt, bev_road.right_lane.sampled_points_m)
                
            if bev_road.centerline and bev_road.centerline.sampled_points_m is not None:
                cx, cz, dist_center = self._find_closest_point_on_curve(v_pt, bev_road.centerline.sampled_points_m)
                # Compute lateral offset by looking at signed X distance to centerline at similar Z
                # Simplified: just use c_x_m - cx
                lateral_offset_m = c_x_m - cx
            else:
                lateral_offset_m = c_x_m # Fallback
                
            overlaps_marking = False
            half_w = v_width / 2.0
            if dist_left < half_w or dist_right < half_w:
                overlaps_marking = True
                
            # Determine Lane ID
            # Center lane is between left and right boundary
            lane_id = "Unknown"
            if dist_center < self.lane_width_m / 2.0:
                lane_id = "Center"
            elif lateral_offset_m < -self.lane_width_m / 2.0:
                lane_id = "Left"
            elif lateral_offset_m > self.lane_width_m / 2.0:
                lane_id = "Right"
                
            assignment = VehicleLaneAssignment(
                vehicle_id=v.vehicle_id if hasattr(v, 'vehicle_id') else vme.vehicle_id,
                lane_id=lane_id,
                lateral_offset_m=lateral_offset_m,
                longitudinal_dist_m=c_z_m,
                dist_to_left_bound_m=dist_left,
                dist_to_right_bound_m=dist_right,
                overlaps_marking=overlaps_marking
            )
            assignments.append(assignment)
            
        return assignments
