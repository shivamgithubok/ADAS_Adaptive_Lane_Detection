import sys
from pathlib import Path
import cv2
import time
import numpy as np
import math

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'phase_01_perception'))
sys.path.append(str(base_dir / 'phase_02_geometry'))

# Phase 1 imports
import config as p1_config
from detector import Detector
from tracker import TrackerHistory
from segmentation import SegmentationAnalyzer
from contact_region import ContactRegionExtractor
from validation import Validator
from utils import format_bbox

# Phase 2 imports
from homography import Homography
from projector import Projector
from footprint_stabilizer import FootprintStabilizer
from metric_scaling import MetricScaler
from dataclasses import dataclass
from typing import Tuple

@dataclass
class VehicleObject:
    track_id: int
    class_name: str
    confidence: float
    ground_left: Tuple[int, int]
    ground_right: Tuple[int, int]

import lane_detection_v0 as ldv0
from lane_projector import LaneProjector
from visualization import LaneVisualizer, DebugFlags
from operational_zone import ODZFilter
from occlusion_manager import OcclusionManager
from event_logger import EventLogger

def run_phase3_engine(video_path: str):
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
    stabilizer = FootprintStabilizer()
    metric_scaler = MetricScaler(homography)
    
    # Initialize Phase 3 Tools
    lane_projector = LaneProjector(homography)
    occlusion_manager = OcclusionManager()
    odz_filter = ODZFilter()
    visualizer = LaneVisualizer()
    event_logger = EventLogger()
    debug_flags = DebugFlags()  # All stages ON by default; toggle any flag here
    debug_flags.SHOW_ROI = False
    debug_flags.SHOW_LANE_LINES = True
    
    print("====================================================")
    print("Phase 3.1: Lane Geometry Pipeline Refactored")
    print("====================================================")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            orig_h, orig_w = frame.shape[:2]
                
            # Run Phase 1 detection
            results = detector.detect_and_track(frame, persist=True)
            detected_objects = []

            if results and results.boxes and results.masks:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None] * len(boxes)
                classes = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                masks = results.masks.data.cpu().numpy()
                
                for i in range(len(boxes)):
                    track_id = track_ids[i]
                    if track_id is None or classes[i] not in p1_config.TARGET_CLASSES:
                        continue
                    
                    bbox = format_bbox(boxes[i])
                    mask_raw = masks[i]
                    mask_img = cv2.resize(mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    mask_img = (mask_img * 255).astype(np.uint8)

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

                if contact_data is not None and status != "FAIL":
                    left_pt = contact_data.get('left_contact', (np.nan, np.nan))
                    right_pt = contact_data.get('right_contact', (np.nan, np.nan))
                    
                    v_obj = VehicleObject(
                        track_id=track_id,
                        class_name=obj['class_name'],
                        confidence=obj['conf'],
                        ground_left=left_pt,
                        ground_right=right_pt
                    )
                
                    vg, _ = projector.process(v_obj, orig_w, orig_h, frame_count)
                    if vg is not None:
                        timestamp = frame_count / 30.0 
                        fg, debug_data = stabilizer.process(
                            vg, obj['mask_img'], homography, orig_w, orig_h, timestamp
                        )
                        if fg is not None:
                            vme = metric_scaler.process(fg, debug_data)
                            vg.footprint = fg
                            vg.metric_estimate = vme
                            geometry_objects.append(vg)

            # PHASE 3.1: Refactored Lane Pipeline
            
            # 1. Lane Detection (lane_detection_v0)
            ld_result, roi_poly, car_boxes, clean_mask, roi_vis, calibrated, left_line, right_line = ldv0.detect_lane_geometry(frame)
            lane_debug_data = {
                'roi_poly': roi_poly,
                'car_boxes': car_boxes,
                'clean_mask': clean_mask,
                'roi_vis': roi_vis,
                'calibrated': calibrated,
                'left_line': left_line,
                'right_line': right_line,
                'ld_result': ld_result
            }
            
            # 2. BEV Projection (lane_projector)
            bev_road = lane_projector.process(ld_result, orig_w, orig_h)
            
            # Phase 3.1.0: Occlusion Management (runs BEFORE ODZ)
            occlusion_states = occlusion_manager.process(
                geometry_objects, detected_objects, orig_w, orig_h
            )
            
            # Phase 3.1.1: ODZ Filter
            odz_states = odz_filter.process(geometry_objects, detected_objects, orig_w, orig_h, homography)
            
            event_logger.update(frame_count, geometry_objects, odz_states, occlusion_states)
            
            # 6. Visualization
            win1 = visualizer.draw_window1(
                frame, bev_road, lane_debug_data, detected_objects,
                odz_states, debug_flags, occlusion_states
            )
            win2 = visualizer.draw_window2(bev_road, odz_states, homography, occlusion_states)
            win3 = visualizer.draw_window3(lane_debug_data, bev_road, occlusion_states)
            
            cv2.imshow("Window 1: Original Image", win1)
            cv2.imshow("Window 2: BEV Footprint", win2)
            cv2.imshow("Window 3: Geometry Panel", win3)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3.1 Lane Geometry Engine")
    parser.add_argument("--video", type=str, default=str(base_dir / "test-video_720.mp4"), help="Path to video file")
    args = parser.parse_args()
    
    run_phase3_engine(args.video)
