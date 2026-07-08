import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import sys
from pathlib import Path
import cv2
import time
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple

base_dir = Path(__file__).resolve().parent.parent


from adas.config.settings import config as p1_config
from adas.perception.vehicle.detector import Detector
from adas.core.tracking.tracker import TrackerHistory
from adas.perception.vehicle.segmentation import SegmentationAnalyzer
from adas.perception.vehicle.contact_region import ContactRegionExtractor
from adas.perception.vehicle.validation import Validator
from adas.common.utils import format_bbox

from adas.core.geometry.homography import Homography
from adas.core.projection.projector import Projector
from adas.visualization.geometry_vis import BEVVisualizer, StabilizationVisualizer
from adas.core.geometry.debug import GeometryDebug, log_vehicle_geometry
from adas.core.geometry.footprint_stabilizer import FootprintStabilizer
from adas.core.geometry.metric_scaling import MetricScaler
from adas.core.geometry.validation_analysis import GeometryValidator

@dataclass
class VehicleObject:
    track_id: int
    class_name: str
    confidence: float
    ground_left: Tuple[int, int]
    ground_right: Tuple[int, int]

def run_geometry_engine(video_path: str, estimator_type: str = "none"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return

    # Initialize Phase 1 Tools
    detector = Detector()
    tracker_history = TrackerHistory(max_history=p1_config.TRACK_MAX_AGE)
    
    # Initialize Phase 2 Tools
    homography = Homography()
    valid_matrix, msg = homography.validate_matrix()
    if not valid_matrix:
        print(f"Homography initialization failed: {msg}")
        return
        
    projector = Projector(homography)
    visualizer = BEVVisualizer()
    debugger = GeometryDebug()
    stabilizer = FootprintStabilizer()
    stab_visualizer = StabilizationVisualizer()
    metric_scaler = MetricScaler(homography)
    from adas.core.geometry.dimension_estimator_interface import DummyGeometryFallback, VehicleObservation
    from adas.core.geometry.estimators.smoke_estimator import SMOKEEstimator
    from adas.core.geometry.estimators.rtm3d_estimator import RTM3DEstimator
    
    if estimator_type == "smoke":
        estimator = SMOKEEstimator()
    elif estimator_type == "rtm3d":
        estimator = RTM3DEstimator()
    else:
        estimator = None
        
    validator = GeometryValidator()
    
    print("====================================================")
    print("Phase 2C Geometry Pipeline")
    print("\nMODE:")
    print("GEOMETRY ONLY")
    print("\nOutputs")
    print("✓ Vehicle Ground Contact")
    print("✓ Vehicle Heading")
    print("✓ Vehicle Footprint")
    print("✓ Vehicle Distance")
    print("✓ BEV Occupancy")
    print("\nNot Estimated")
    print("✗ Vehicle Height")
    print("✗ Vehicle Roof")
    print("✗ Vehicle 3D Shape")
    print("✗ Vehicle Cuboid")
    print("====================================================")
    
    # Global Flag to Disable Image Cuboids in Geometry Only Mode
    ENABLE_IMAGE_CUBOID = False
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
                
            # Run Phase 1 detection
            results = detector.detect_and_track(frame, persist=True)
            detected_objects = []

            if results and results.boxes and results.masks:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None] * len(boxes)
                classes = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                masks = results.masks.data.cpu().numpy()
                orig_h, orig_w = frame.shape[:2]
                if frame_count % 30 == 0:
                    print("\nFrame:")
                    print(f"{orig_w}x{orig_h}")
                    print("Calibration:")
                    print(f"{homography.calib_width}x{homography.calib_height}")
                    print("Scale:")
                    print(f"{homography.calib_width / orig_w:.3f}")
                    print(f"{homography.calib_height / orig_h:.3f}\n")

                for i in range(len(boxes)):
                    track_id = track_ids[i]
                    if track_id is None or classes[i] not in p1_config.TARGET_CLASSES:
                        continue
                    
                    bbox = format_bbox(boxes[i])
                    mask_raw = masks[i]
                    mask_raw_h, mask_raw_w = mask_raw.shape[:2]
                    mask_img = cv2.resize(mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    mask_img = (mask_img * 255).astype(np.uint8)
                    mask_img_h, mask_img_w = mask_img.shape[:2]

                    obj_data = {
                        'track_id': track_id,
                        'class_name': p1_config.CLASS_NAMES.get(classes[i], "Unknown"),
                        'conf': confs[i],
                        'bbox': bbox,
                        'mask_img': mask_img
                    }
                    detected_objects.append(obj_data)

            tracker_history.update(detected_objects)
            geometry_objects = []

            for obj in detected_objects:
                mask_img = obj['mask_img']
                bbox = obj['bbox']
                track_id = obj['track_id']

                seg_data = SegmentationAnalyzer.process_mask(mask_img)
                contact_data = ContactRegionExtractor.extract(mask_img, bbox)
                track_info = tracker_history.get_track_info(track_id)
                score, status, reason = Validator.validate(obj, seg_data, contact_data, track_info)

                # Step 2: Debug Ground Contact Extraction
                mask_h = contact_data.get('mask_height', 0) if contact_data else 0
                strip_h = contact_data.get('strip_height', 0) if contact_data else 0
                g_px = contact_data.get('ground_pixels', 0) if contact_data else 0
                g_w = contact_data.get('ground_width', 0) if contact_data else 0
            
                # Step 3: Visualize Ground Contact
                if contact_data is not None:
                    left_pt = contact_data.get('left_contact', (np.nan, np.nan))
                    right_pt = contact_data.get('right_contact', (np.nan, np.nan))
                    center_pt = contact_data.get('median_contact', (np.nan, np.nan))
                
                    if not math.isnan(left_pt[0]):
                        cv2.circle(frame, left_pt, 5, (0, 0, 255), -1) # Red (Left)
                    if not math.isnan(right_pt[0]):
                        cv2.circle(frame, right_pt, 5, (255, 0, 0), -1) # Blue (Right)
                    if not math.isnan(center_pt[0]):
                        cv2.circle(frame, center_pt, 5, (255, 255, 0), -1) # Cyan (Center)
                    
                    if status != "FAIL":
                        v_obj = VehicleObject(
                            track_id=track_id,
                            class_name=obj['class_name'],
                            confidence=obj['conf'],
                            ground_left=left_pt,
                            ground_right=right_pt
                        )
                    
                        vg, reject_reason = projector.process(v_obj, orig_w, orig_h, frame_count)
                        if vg is not None:
                            # NEW: Geometry Stabilization Phase 2B.1
                            timestamp = frame_count / 30.0 # Approximate
                            fg, debug_data = stabilizer.process(
                                vg, obj['mask_img'], homography, orig_w, orig_h, timestamp
                            )
                            if fg is not None:
                                # Metric Scaling (Phase 2C.1.5)
                                vme = metric_scaler.process(fg, debug_data)
                                
                                # Deep Dimension Estimation (Phase 2C.2)
                                dim_estimate = None
                                if estimator is not None:
                                    obs = VehicleObservation(
                                        vehicle_id=fg.vehicle_id,
                                        image_crop=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                                        bounding_box_2d=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                        geometric_distance_m=vme.distance_m,
                                        geometric_heading_deg=debug_data.get("heading", 0.0),
                                        class_id=classes[i]
                                    )
                                    dim_estimate = estimator.estimate_dimensions(obs)
                                    
                                    # Override metric estimate width/length for downstream logic and visualization
                                    if dim_estimate is not None:
                                        # Keep the geometric distance but use learned dimensions
                                        vme.corrected_width_m = dim_estimate.width_m
                                        vme.corrected_length_m = dim_estimate.length_m
                            
                                # Validate (Phase 2C.1.6)
                                validator.add_observation(vme, fg, debug_data)
                            
                                # Attach the footprint geometry to the vehicle geometry or use it
                                vg.footprint = fg
                                vg.metric_estimate = vme
                                vg.dim_estimate = dim_estimate # Keep track of the deep estimate separately
                                geometry_objects.append(vg)
                            
                                # For visualization
                                if not hasattr(stabilizer, 'debug_list'):
                                    stabilizer.debug_list = []
                                stabilizer.debug_list.append((fg, debug_data, vme, dim_estimate))
                            else:
                                # Fallback if stabilization fails but projection passed (should rarely happen)
                                geometry_objects.append(vg)
                            
            # Step 7 & 8: BEV Renderer & Debug Window
            bev_frame = visualizer.render(
                geometry_objects,
                frame_w=orig_w,
                frame_h=orig_h,
                calib_w=homography.calib_width,
                calib_h=homography.calib_height
            )
            debug_frame = debugger.render(geometry_objects)
        
            # Geometry Stabilization Debug Render
            bev_canvas = np.zeros((800, 800, 3), dtype=np.uint8)
            panel_canvas = np.zeros((800, 400, 3), dtype=np.uint8)
        
            if hasattr(stabilizer, 'debug_list') and len(stabilizer.debug_list) > 0:
            
                # Draw strips on Window 1 (Original Frame) for ALL vehicles
                for fg, debug_data, vme, dim_est in stabilizer.debug_list:
                    if debug_data is None: continue
                    for s in debug_data["strip_candidates"]:
                        is_acc = any(a['id'] == s['id'] for a in debug_data["accepted_strips"])
                        color = (0, 255, 0) if is_acc else (0, 0, 255)
                        # Draw the strip line
                        pt1 = (int(s['left'][0]), int(s['left'][1]))
                        pt2 = (int(s['right'][0]), int(s['right'][1]))
                        cv2.line(frame, pt1, pt2, color, 2)
                        # Draw ground pixels (small blue dots) on the frame
                        y_start, y_end = s['y_start'], s['y_end']
                        mask = debug_data["mask"]
                        strip_y, strip_x = np.where(mask[y_start:y_end, :] > 128)
                        for px, py in zip(strip_x, strip_y + y_start):
                            cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)
                            
                    # --- NEW: Draw 3D Bounding Box on Original Image ---
                    if ENABLE_IMAGE_CUBOID and vme is not None and vme.corrected_polygon is not None:
                        height_m = dim_est.height_m if dim_est else 1.5
                        
                        bottom_pts = []
                        top_pts = []
                        valid_box = True
                        
                        for x_px, y_px in vme.corrected_polygon:
                            ub, vb = homography.project_3d_to_image(x_px, y_px, 0.0)
                            ut, vt = homography.project_3d_to_image(x_px, y_px, height_m)
                            if ub == float('inf') or ut == float('inf'):
                                valid_box = False
                                break
                            bottom_pts.append((int(ub), int(vb)))
                            top_pts.append((int(ut), int(vt)))
                            
                        if valid_box:
                            color_3d = (0, 255, 255) if dim_est else (255, 255, 255) # Yellow if learned, White if geometry
                            
                            # Draw bottom footprint
                            cv2.polylines(frame, [np.array(bottom_pts, dtype=np.int32)], True, color_3d, 2)
                            # Draw top face
                            cv2.polylines(frame, [np.array(top_pts, dtype=np.int32)], True, color_3d, 2)
                            # Draw vertical pillars
                            for i in range(4):
                                cv2.line(frame, bottom_pts[i], top_pts[i], color_3d, 2)
                                
                    # Task 7: Per Vehicle Validation
                    print(f"Vehicle {fg.vehicle_id}")
                    print(f"Ground Contact      ✓")
                    print(f"Projection          ✓")
                    print(f"Heading             ✓")
                    print(f"Footprint           ✓")
                    print(f"Metric Conversion   ✓")
                    if not ENABLE_IMAGE_CUBOID:
                        print(f"Image Cuboid        DISABLED")
                        print(f"Reason:")
                        print(f"Requires learned monocular 3D estimation.\n")
            
                # Render Window 2 (BEV) and Window 3 (Panel)
                bev_canvas, panel_canvas = stab_visualizer.render(stabilizer.debug_list)
            
                stabilizer.debug_list = [] # Reset for next frame
        
            cv2.imshow("Window 1: Original Image", frame)
            cv2.imshow("Window 2: BEV Footprint", bev_canvas)
            cv2.imshow("Window 3: Geometry Panel", panel_canvas)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        validator.generate_report()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2A-2C Geometry Engine")
    parser.add_argument("--video", type=str, default=str(base_dir / "test-video_720.mp4"), help="Path to video file")
    parser.add_argument("--estimator", type=str, choices=["none", "smoke", "rtm3d"], default="none", help="Dimension estimator to use")
    args = parser.parse_args()
    
    run_geometry_engine(args.video, estimator_type=args.estimator)
