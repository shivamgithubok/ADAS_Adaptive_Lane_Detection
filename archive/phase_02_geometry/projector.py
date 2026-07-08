import math
from vehicle_geometry import VehicleGeometry

CLASS_LENGTH_PRIORS = {
    "Car": 4.5,
    "SUV": 4.8,
    "Truck": 10.0,
    "Bus": 12.0,
    "Motorcycle": 2.2
}

class Projector:
    def __init__(self, homography):
        self.homography = homography

    def process(self, v_obj, frame_w, frame_h, frame_idx):
        """
        Takes a VehicleObject from Phase 01.
        Returns a validated VehicleGeometry object, or (None, reason) if rejected.
        """
        calib_w = self.homography.calib_width
        calib_h = self.homography.calib_height
        sx = calib_w / frame_w
        sy = calib_h / frame_h
        
        # Step 3: Print scaling info once every 30 frames
        if frame_idx % 30 == 0 and v_obj.track_id == 1:
            pass
        # Step 2: Project Ground Contact with scaling
        # Ensure left is actually left
        orig_left = v_obj.ground_left
        orig_right = v_obj.ground_right
        
        if orig_left[0] > orig_right[0]:
            orig_left, orig_right = orig_right, orig_left
            
        orig_center = (
            int((orig_left[0] + orig_right[0]) / 2),
            int((orig_left[1] + orig_right[1]) / 2)
        )
        
        left_pt = (orig_left[0] * sx, orig_left[1] * sy)
        right_pt = (orig_right[0] * sx, orig_right[1] * sy)
        center_pt = (orig_center[0] * sx, orig_center[1] * sy)

        # Step 4: Debug print removed

        # Step 2 & 3: Homography and Raw coordinates
        raw_l = self.homography.project_point_raw(left_pt[0], left_pt[1])
        raw_r = self.homography.project_point_raw(right_pt[0], right_pt[1])
        raw_c = self.homography.project_point_raw(center_pt[0], center_pt[1])

        bev_left = self.homography.project_point(left_pt[0], left_pt[1])
        bev_right = self.homography.project_point(right_pt[0], right_pt[1])
        bev_center_img = self.homography.project_point(center_pt[0], center_pt[1])
        
        # Step 4: Validate every projected point
        reject_reason = None
        for name, raw_pt, bev_pt in [("Left", raw_l, bev_left), ("Center", raw_c, bev_center_img), ("Right", raw_r, bev_right)]:
            if math.isnan(bev_pt[0]) or math.isnan(bev_pt[1]):
                reject_reason = f"{name} point is NaN"
                break
            if math.isinf(bev_pt[0]) or math.isinf(bev_pt[1]):
                reject_reason = f"{name} point is Inf"
                break
            # Step 6: Removed "negative homogeneous scale" rejection!
            if bev_pt[0] < -5000 or bev_pt[0] > 5000 or bev_pt[1] < -5000 or bev_pt[1] > 5000:
                reject_reason = f"{name} point outside extreme BEV bounds"
                break

        if reject_reason:
            proj_width_px = 0.0
            width_px = abs(right_pt[0] - left_pt[0])
            bev_center_metric = (0.0, 0.0)
        else:
            proj_width_px = math.hypot(bev_right[0] - bev_left[0], bev_right[1] - bev_left[1])
            width_px = abs(right_pt[0] - left_pt[0])
            cx_m = (bev_center_img[0] - 400.0) * self.homography.MPP
            cz_m = (800.0 - bev_center_img[1]) * self.homography.MPP
            bev_center_metric = (cx_m, cz_m)

        validation_status = "Rejected" if reject_reason else "Valid"
        
        vg = VehicleGeometry(
            id=v_obj.track_id,
            class_name=v_obj.class_name,
            confidence=v_obj.confidence,
            img_left=left_pt,
            img_right=right_pt,
            img_center=center_pt,
            bev_left=bev_left,
            bev_right=bev_right,
            bev_center=bev_center_img,
            width_px=int(proj_width_px),
            width_m=0.0,
            length_m=0.0,
            footprint_fl=(0,0),
            footprint_fr=(0,0),
            footprint_rl=(0,0),
            footprint_rr=(0,0)
        )
        vg.validation_status = validation_status
        return vg, reject_reason
