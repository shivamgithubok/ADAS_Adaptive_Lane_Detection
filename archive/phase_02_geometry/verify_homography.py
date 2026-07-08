import sys
from pathlib import Path
import cv2
import numpy as np
import math

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / 'phase_01_perception'))
from homography import Homography

def verify_homography():
    print("--- Phase 2A Homography Verification ---")
    homography = Homography()

    print("\n## Step 1: Matrix Creation")
    print("Source Points: Implicitly modeled via Pinhole Flat-Earth logic")
    print("Destination Points: Implicit BEV Canvas mapping")
    print("H Matrix:")
    print(homography.H)
    print("\nH Inverse (Image -> BEV Matrix):")
    print(homography.H_inv)

    print("\n## Step 2: H Representation")
    print("H represents: BEV -> Image")
    print("H_inv represents: Image -> BEV")

    print("\n## Step 3: Transformation Uses")
    print("cv2.perspectiveTransform() and `H_inv @ point` use: H inverse (which we defined as Image->BEV)")

    print("\n## Step 4: Reconstruct Point")
    u, v = 480, 470  # Picked v=470 because v must be > cy (360) for ground contact
    bev = homography.project_point(u, v)
    rec = homography.inverse_project(bev[0], bev[1])
    
    print(f"Original: ({u}, {v})")
    print(f"Projected: ({bev[0]:.2f}, {bev[1]:.2f})")
    print(f"Reconstructed: ({rec[0]:.2f}, {rec[1]:.2f})")

    print("\n## Step 5: Homogeneous Coordinates")
    raw = homography.project_point_raw(u, v)
    print("Before normalization:")
    print(f"({raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f})")
    print("After normalization:")
    print(f"({bev[0]:.2f}, {bev[1]:.2f})")
    print("w is expected to be POSITIVE (w = v - cy).")

    print("\n## Step 6: Draw Reference Points")
    print("Drawing visually in verify_vis.png...")
    
    # Render visual
    img = np.zeros((800, 1280, 3), dtype=np.uint8)
    bev_img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    pts = [(640, 800), (400, 500), (880, 500), (640, 400)]
    for pt in pts:
        cv2.circle(img, pt, 5, (0,0,255), -1)
        b = homography.project_point(*pt)
        cv2.circle(bev_img, (int(b[0]), int(b[1])), 5, (0,0,255), -1)
        
    cv2.imwrite("verify_img.png", img)
    cv2.imwrite("verify_bev.png", bev_img)

    print("\n## Step 7: Preserve Left-Right Ordering")
    L = homography.project_point(400, 500)
    R = homography.project_point(880, 500)
    print(f"Image: Left x (400) < Right x (880)")
    print(f"BEV: Left x ({L[0]:.1f}) {'<' if L[0] < R[0] else '>'} Right x ({R[0]:.1f})")

    print("\n## Step 8: Depth Collapse")
    near = homography.project_point(640, 800)
    far = homography.project_point(640, 400)
    print(f"Near Y (v=800): {near[1]:.1f}")
    print(f"Far Y (v=400): {far[1]:.1f}")
    
if __name__ == "__main__":
    verify_homography()
