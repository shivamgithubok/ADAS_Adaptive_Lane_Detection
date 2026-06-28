import yaml
import numpy as np
from pathlib import Path

class Homography:
    def __init__(self, calib_path=None):
        if calib_path is None:
            # Default to the project config
            base_dir = Path(__file__).resolve().parent.parent
            calib_path = base_dir / "config" / "camera_calib.yaml"
            
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)
            
        self.fx = calib.get('fx', 800.0)
        self.fy = calib.get('fy', 800.0)
        self.cx = calib.get('cx', 640.0)
        self.cy = calib.get('cy', 360.0)
        self.height = calib.get('height', 1.2)
        self.pitch = calib.get('pitch', 0.0)  # In radians or degrees? We assume radians, but it's 0.0.
        self.calib_width = calib.get('calibration_width', 1280)
        self.calib_height = calib.get('calibration_height', 720)

        self.PPM = 145.0 / 3.70  # Pixels Per Meter
        self.MPP = 3.70 / 145.0  # Meters Per Pixel
        self._build_matrix()

    def _build_matrix(self):
        # Maps (u, v) to metric (X, Z) then to BEV Pixel Canvas (x_px, y_px)
        # X_bev = (u - cx) * height / (v - cy) * (fy / fx)
        # Z_bev = height * fy / (v - cy)
        # We want:
        # x_px = 400 + X_bev * PPM
        # y_px = 800 - Z_bev * PPM
        
        h_fy_fx = (self.height * self.fy) / self.fx
        h_fy = self.height * self.fy
        
        # Metric mapping
        # [ X_bev * W ]   [ (height * fy / fx)      0                   -cx * (height * fy / fx) ] [ u ]
        # [ Z_bev * W ] = [ 0                       0                    height * fy             ] [ v ]
        # [ W         ]   [ 0                       1                   -cy                      ] [ 1 ]
        
        # Now pixel mapping
        # x_px * W = 400 * W + PPM * (X_bev * W)
        # y_px * W = 800 * W - PPM * (Z_bev * W)
        
        row1 = [self.PPM * h_fy_fx, 400.0, -self.PPM * self.cx * h_fy_fx - 400.0 * self.cy]
        row2 = [0.0, 800.0, -self.PPM * h_fy - 800.0 * self.cy]
        row3 = [0.0, 1.0, -self.cy]
        
        self.H_inv = np.array([row1, row2, row3], dtype=np.float32)
        
        try:
            self.H = np.linalg.inv(self.H_inv)
        except np.linalg.LinAlgError:
            self.H = None

    def validate_matrix(self):
        if self.H_inv is None:
            return False, "Matrix failed to initialize."
        if self.H_inv[2, 1] == 0 and self.H_inv[2, 2] == 0:
            return False, "Matrix denominator is zero."
        return True, "Valid Homography Matrix"

    def project_point_raw(self, u, v):
        """Returns raw homogeneous coordinates (x, y, w) before normalization."""
        pt = np.array([u, v, 1.0], dtype=np.float32)
        raw = self.H_inv @ pt
        return tuple(raw)

    def project_point(self, u, v):
        """Projects a single image point (u,v) to BEV (x,y)."""
        raw = self.project_point_raw(u, v)
        if raw[2] == 0:
            return (float('inf'), float('inf'))
        return (raw[0] / raw[2], raw[1] / raw[2])

    def project_points(self, pts):
        """Projects an array of points from Image to BEV."""
        import cv2
        if len(pts) == 0:
            return np.array([])
        # pts should be shape (N, 1, 2)
        if pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)
            
        bev_pts = cv2.perspectiveTransform(pts, self.H_inv)
        return bev_pts

    def inverse_project(self, x, y):
        """Projects BEV (x,y) back to Image (u,v)."""
        import cv2
        if self.H is None:
            return None
        pts = np.array([[[x, y]]], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(pts, self.H)
        return tuple(img_pts[0, 0])
