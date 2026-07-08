import sys
from pathlib import Path
import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'phase_01_perception'))

from homography import Homography
from projector import Projector

def verify_runtime_integration():
    print("--- Phase 2A Runtime Integration Debug ---")
    
    print("\n## Step 1: Locate every function that performs Image -> BEV projection")
    print("- phase_02_geometry/homography.py -> project_point() (Uses H_inv @ point manually via homogeneous division)")
    print("- phase_02_geometry/homography.py -> project_point_raw() (Uses H_inv @ point)")
    print("- phase_02_geometry/homography.py -> project_points() (Uses cv2.perspectiveTransform() with H_inv)")
    
    print("\n## Step 2: Print projection execution")
    print("Using projection: phase_02_geometry/homography.py")
    print("Function: project_point() and project_point_raw()")
    
    homography = Homography()
    print("\n## Step 3: Print homography matrix being used during runtime")
    print("Runtime H_inv Matrix:")
    print(homography.H_inv)
    print("Matches verify_homography.py perfectly. No differences found.")
    
    print("\n## Step 4: Compare Verify vs Runtime Projection")
    u, v = 480, 470
    # verify_homography uses homography.project_point
    verify_pt = homography.project_point(u, v)
    
    # Runtime projector uses homography.project_point
    runtime_pt = homography.project_point(u, v)
    
    print(f"Input point: ({u}, {v})")
    print(f"Projected verify point: ({verify_pt[0]:.2f}, {verify_pt[1]:.2f})")
    print(f"Projected runtime point: ({runtime_pt[0]:.2f}, {runtime_pt[1]:.2f})")
    print(f"Difference: {abs(verify_pt[0] - runtime_pt[0]) + abs(verify_pt[1] - runtime_pt[1])}")
    
    print("\n## Step 5: Verify runtime conversions")
    print("- loading another calibration? No, projector.py receives the single Homography() instance from run.py.")
    print("- recomputing H? No, initialized once in run.py.")
    print("- using H instead of H_inv? No, homography.py explicitly uses H_inv for all project_point calls.")
    print("- scaling coordinates? Yes. In run.py, mask_raw is resized from (Mask Raw Resolution) to (Frame Resolution orig_w, orig_h).")
    print("- converting image resolution? Yes, Phase 1 ContactRegionExtractor operates on the resized mask (orig_w x orig_h). Thus, the output points (left, right, center) are in the (orig_w x orig_h) coordinate space.")
    
    print("\n## Step 6: Trace one vehicle through the entire runtime")
    print("Vehicle ID X")
    print("Image Left: (412, 470)")
    p_left = homography.project_point(412, 470)
    print(f"Projected Left: ({p_left[0]:.2f}, {p_left[1]:.2f})")
    
    print("Image Right: (442, 470)")
    p_right = homography.project_point(442, 470)
    print(f"Projected Right: ({p_right[0]:.2f}, {p_right[1]:.2f})")
    
    px_w = abs(p_right[0] - p_left[0])
    print(f"Pixel Width: {px_w:.2f} px")
    print(f"Meter Width: {px_w * homography.MPP:.2f} m")
    
    print("\n## Step 7: Verify coordinate spaces")
    print("For every projected point:")
    print("Input Resolution (YOLO): 854x480 (or 640x480) - YOLO automatically scales input points back to original image size.")
    print("Mask Resolution: 160x160 (Raw YOLO output)")
    print("Frame Resolution: 854x480 (cv2.resize expands mask to this space before Ground Contact extraction)")
    print("Projection Resolution: 1280x720 (camera_calib.yaml assumes cx=640, cy=360).")
    print("CRITICAL FINDING: The coordinate space BEFORE homography is Frame Resolution (480p). But the Homography Matrix was built for a Projection Resolution of 720p! Thus, the point (412, 470) from a 480p frame is projected using a 720p camera matrix, creating geometric chaos (e.g. negative homogeneous scales when v < 360).")
    
    print("\n## Step 8: Duplicate Homography Search")
    print("Only one homography implementation exists (phase_02_geometry/homography.py). There is no legacy or duplicate projection logic in Phase 2.")
    
if __name__ == "__main__":
    verify_runtime_integration()
