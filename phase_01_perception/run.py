import argparse
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List, Any

import config
from detector import Detector
from tracker import TrackerHistory
from segmentation import SegmentationAnalyzer
from contact_region import ContactRegionExtractor
from profiler import Profiler
from visualization import Visualizer
from utils import format_bbox
from validation import Validator

@dataclass
class VehicleObject:
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    segmentation_mask: np.ndarray
    ground_left: Tuple[int, int]
    ground_right: Tuple[int, int]
    ground_width: int
    mask_area: int
    track_age: int
    timestamp: float

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return

    detector = Detector()
    tracker_history = TrackerHistory(max_history=config.TRACK_MAX_AGE)
    profiler = Profiler()

    frame_count = 0
    debug_mode = 0 # 0=Clean/Production, 1=Det, 2=Seg, 3=Contact, 4=Track, 5=Prof, 6=All
    
    stats_history = {
        'detected': 0, 'passed': 0, 'low_conf': 0, 'failed': 0,
        'strip_heights': [], 'ground_widths': [], 'ground_pixels': [], 'scores': []
    }

    while cap.isOpened():
        profiler.start("Video Read")
        ret, frame = cap.read()
        profiler.stop("Video Read")
        
        if not ret:
            break
            
        frame_count += 1
        
        profiler.start("YOLO")
        results = detector.detect_and_track(frame, persist=True)
        profiler.stop("YOLO")

        detected_objects = []

        if results and results.boxes and results.masks:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None] * len(boxes)
            classes = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            
            masks = results.masks.data.cpu().numpy()
            orig_h, orig_w = frame.shape[:2]

            for i in range(len(boxes)):
                track_id = track_ids[i]
                cls_id = classes[i]
                conf = confs[i]
                
                if cls_id not in config.TARGET_CLASSES:
                    continue

                bbox = format_bbox(boxes[i])

                obj_data = {
                    'track_id': track_id,
                    'class_id': cls_id,
                    'class_name': config.CLASS_NAMES.get(cls_id, "Unknown"),
                    'conf': conf,
                    'bbox': bbox,
                }
                
                mask_raw = masks[i]
                mask_img = cv2.resize(mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask_img = (mask_img * 255).astype(np.uint8)
                obj_data['mask_img'] = mask_img

                detected_objects.append(obj_data)

        profiler.start("Tracking")
        tracker_history.update(detected_objects)
        profiler.stop("Tracking")

        # Gather exported objects for Phase 02
        exported_vehicles = []

        for obj in detected_objects:
            mask_img = obj['mask_img']
            bbox = obj['bbox']
            track_id = obj['track_id']

            profiler.start("Segmentation")
            seg_data = SegmentationAnalyzer.process_mask(mask_img)
            profiler.stop("Segmentation")

            profiler.start("Ground Contact")
            contact_data = ContactRegionExtractor.extract(mask_img, bbox)
            profiler.stop("Ground Contact")

            track_info = tracker_history.get_track_info(track_id)
            
            # Validation
            score, status, reason = Validator.validate(obj, seg_data, contact_data, track_info)
            
            # Record stats
            stats_history['detected'] += 1
            if status == "PASS":
                stats_history['passed'] += 1
            elif status == "LOW_CONFIDENCE":
                stats_history['low_conf'] += 1
            else:
                stats_history['failed'] += 1
                
            if contact_data is not None:
                stats_history['strip_heights'].append(contact_data.get('strip_height', 0))
                stats_history['ground_widths'].append(contact_data.get('ground_width', 0))
                stats_history['ground_pixels'].append(contact_data.get('ground_pixels', 0))
            stats_history['scores'].append(score)

            # Debug Information Output
            if debug_mode != 0:
                mask_h = contact_data.get('mask_height', 0) if contact_data else 0
                strip_h = contact_data.get('strip_height', 0) if contact_data else 0
                g_px = contact_data.get('ground_pixels', 0) if contact_data else 0
                g_w = contact_data.get('ground_width', 0) if contact_data else 0
                print(f"ID={track_id} | MaskH={mask_h} px | Strip={strip_h} px | GroundPx={g_px} | Width={g_w} px | Score={score} {status}" + (f" ({reason})" if reason else ""))

            if status == "FAIL":
                if debug_mode != 0:
                    print(f"Vehicle {track_id}")
                    print("Ground Contact Failed")
                    print("Reason:")
                    print(reason)
                    print(f"Pixels = {contact_data.get('ground_pixels', 0) if contact_data else 0}")
                    print(f"Strip = {contact_data.get('strip_height', 0) if contact_data else 0} px")
                    print(f"Quality = {score}")
                    print("")
                continue

            # Export Object
            v_obj = VehicleObject(
                track_id=track_id,
                class_name=obj['class_name'],
                confidence=obj['conf'],
                bbox=bbox,
                segmentation_mask=mask_img,
                ground_left=contact_data.get('left_contact', (0,0)) if contact_data else (0,0),
                ground_right=contact_data.get('right_contact', (0,0)) if contact_data else (0,0),
                ground_width=contact_data.get('ground_width', 0) if contact_data else 0,
                mask_area=seg_data.get('area', 0),
                track_age=track_info['age'] if track_info else 0,
                timestamp=time.time()
            )
            exported_vehicles.append(v_obj)

            obj['score'] = score
            obj['status'] = status
            
            profiler.start("Drawing")
            frame = Visualizer.draw(frame, obj, seg_data, contact_data, track_info, debug_mode)
            profiler.stop("Drawing")

        profiler.start("Drawing")
        if debug_mode in (5, 6):
            fps = profiler.fps_history[-1] if profiler.fps_history else 0.0
            frame = Visualizer.draw_fps(frame, fps)
        if debug_mode != 0:
            frame = Visualizer.draw_debug_overlay(frame, stats_history, len(detected_objects))
        profiler.stop("Drawing")

        profiler.step()
        if debug_mode in (5, 6) and profiler.should_print():
            profiler.print_profile()
            
        if frame_count % 30 == 0:
            print("\n--- 30-Frame Statistics ---")
            print(f"Vehicles Detected: {stats_history['detected']}")
            print(f"Vehicles Passed: {stats_history['passed']}")
            print(f"Low Confidence: {stats_history['low_conf']}")
            print(f"Failed: {stats_history['failed']}")
            avg_strip = sum(stats_history['strip_heights']) / len(stats_history['strip_heights']) if stats_history['strip_heights'] else 0
            avg_gw = sum(stats_history['ground_widths']) / len(stats_history['ground_widths']) if stats_history['ground_widths'] else 0
            avg_gp = sum(stats_history['ground_pixels']) / len(stats_history['ground_pixels']) if stats_history['ground_pixels'] else 0
            avg_score = sum(stats_history['scores']) / len(stats_history['scores']) if stats_history['scores'] else 0
            print(f"Average Strip Height: {avg_strip:.1f} px")
            print(f"Average Ground Width: {avg_gw:.1f} px")
            print(f"Average Ground Pixels: {avg_gp:.1f}")
            print(f"Average Quality Score: {avg_score:.1f}")
            print("---------------------------\n")
            stats_history = {
                'detected': 0, 'passed': 0, 'low_conf': 0, 'failed': 0,
                'strip_heights': [], 'ground_widths': [], 'ground_pixels': [], 'scores': []
            }

        cv2.imshow("Phase 01 Perception", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'): debug_mode = 1
        elif key == ord('2'): debug_mode = 2
        elif key == ord('3'): debug_mode = 3
        elif key == ord('4'): debug_mode = 4
        elif key == ord('5'): debug_mode = 5
        elif key == ord('6'): debug_mode = 6
        elif key == ord('0'): debug_mode = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 01 Perception Pipeline")
    parser.add_argument("--video", type=str, default="../test-video_480.mp4", help="Path to input video")
    args = parser.parse_args()
    
    process_video(args.video)
