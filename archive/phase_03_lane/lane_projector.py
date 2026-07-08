import numpy as np
from typing import Optional, Tuple
from lane_geometry import LaneCurve, RoadGeometry

class LaneProjector:
    def __init__(self, homography):
        self.homography = homography
        self.PPM = self.homography.PPM
        self.lane_id_counter = 0

    def _create_lane_curve(self, image_poly, sampled_image_points, id_val: int, frame_w: int, frame_h: int) -> Optional[LaneCurve]:
        if sampled_image_points is None or len(sampled_image_points) == 0:
            return None
        
        calib_w = self.homography.calib_width
        calib_h = self.homography.calib_height
        sx = calib_w / frame_w
        sy = calib_h / frame_h
        
        # Scale points to calibration resolution
        scaled_pts = sampled_image_points.copy()
        scaled_pts[:, 0] *= sx
        scaled_pts[:, 1] *= sy
        
        # Project all points
        pts_img = np.array([scaled_pts], dtype=np.float32)
        pts_bev = self.homography.project_points(pts_img).reshape(-1, 2)
        
        # Fit degree 2 polynomial in BEV: x = Ay^2 + By + C
        bev_y = pts_bev[:, 1]
        bev_x = pts_bev[:, 0]
        
        try:
            poly_coeffs = np.polyfit(bev_y, bev_x, 2)
        except Exception:
            poly_coeffs = np.array([0.0, 0.0, bev_x[0]])
            
        # Metric conversion
        A_m = poly_coeffs[0] * self.PPM
        B_m = poly_coeffs[1]
        C_m = poly_coeffs[2] / self.PPM
        poly_coeffs_m = np.array([A_m, B_m, C_m])
        
        start_y = 600.0
        end_y = 800.0
        
        ploty = np.linspace(start_y, end_y, num=100)
        plotx = poly_coeffs[0]*ploty**2 + poly_coeffs[1]*ploty + poly_coeffs[2]
        sampled_points = np.vstack((plotx, ploty)).T
        
        # Curvature at y = 400 (mid)
        y_eval = 400.0
        y_eval_m = y_eval / self.PPM
        
        A = poly_coeffs_m[0]
        B = poly_coeffs_m[1]
        
        if abs(A) < 1e-6:
            curvature_m = 10000.0
            model_type = "STRAIGHT"
        else:
            curvature_m = ((1 + (2*A*y_eval_m + B)**2)**1.5) / abs(2*A)
            if curvature_m > 3000.0:
                model_type = "STRAIGHT"
            elif A > 0:
                model_type = "RIGHT_CURVE"
            else:
                model_type = "LEFT_CURVE"
                
        heading = float(np.arctan(2*A*y_eval_m + B))
        
        return LaneCurve(
            id=id_val,
            image_polynomial=image_poly,
            bev_polynomial=poly_coeffs,
            poly_coeffs_m=poly_coeffs_m,
            image_points=sampled_image_points,
            bev_points=pts_bev,
            sampled_points=sampled_points,
            curvature_m=curvature_m,
            heading=heading,
            visibility=100.0,
            confidence=100.0,
            model_type=model_type
        )

    def process(self, ld_result, frame_w: int, frame_h: int) -> RoadGeometry:
        self.lane_id_counter += 1
        
        left_curve = self._create_lane_curve(ld_result.left_polynomial, ld_result.sampled_image_points_left, self.lane_id_counter, frame_w, frame_h)
        right_curve = self._create_lane_curve(ld_result.right_polynomial, ld_result.sampled_image_points_right, self.lane_id_counter + 1, frame_w, frame_h)
        
        centerline = None
        lane_width = 0.0
        
        if left_curve and right_curve:
            center_coeffs = (left_curve.bev_polynomial + right_curve.bev_polynomial) / 2.0
            center_coeffs_m = (left_curve.poly_coeffs_m + right_curve.poly_coeffs_m) / 2.0
            
            start_y = 600.0
            end_y = 800.0
            
            ploty = np.linspace(start_y, end_y, num=100)
            plotx = center_coeffs[0]*ploty**2 + center_coeffs[1]*ploty + center_coeffs[2]
            sampled_points = np.vstack((plotx, ploty)).T
            
            y_eval = 400.0
            y_eval_m = y_eval / self.PPM
            A = center_coeffs_m[0]
            B = center_coeffs_m[1]
            
            if abs(A) < 1e-6:
                curvature_m = 10000.0
                model_type = "STRAIGHT"
            else:
                curvature_m = ((1 + (2*A*y_eval_m + B)**2)**1.5) / abs(2*A)
                if curvature_m > 3000.0:
                    model_type = "STRAIGHT"
                elif A > 0:
                    model_type = "RIGHT_CURVE"
                else:
                    model_type = "LEFT_CURVE"
                    
            heading = float(np.arctan(2*A*y_eval_m + B))
            
            centerline = LaneCurve(
                id=0,
                image_polynomial=None,
                bev_polynomial=center_coeffs,
                poly_coeffs_m=center_coeffs_m,
                image_points=None,
                bev_points=None,
                sampled_points=sampled_points,
                curvature_m=curvature_m,
                heading=heading,
                visibility=100.0,
                confidence=100.0,
                model_type=model_type
            )
            
            # Estimate lane width in meters
            y_eval = end_y
            xl = left_curve.bev_polynomial[0]*y_eval**2 + left_curve.bev_polynomial[1]*y_eval + left_curve.bev_polynomial[2]
            xr = right_curve.bev_polynomial[0]*y_eval**2 + right_curve.bev_polynomial[1]*y_eval + right_curve.bev_polynomial[2]
            lane_width = abs(xr - xl) / self.PPM
            
        return RoadGeometry(
            left_lane=left_curve,
            right_lane=right_curve,
            centerline=centerline,
            lane_width=lane_width
        )
