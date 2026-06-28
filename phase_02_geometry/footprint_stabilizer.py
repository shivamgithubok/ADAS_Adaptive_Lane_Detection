import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class FootprintGeometry:
    vehicle_id: int
    footprint_points: np.ndarray
    left_point: tuple
    center_point: tuple
    right_point: tuple
    orientation_vector: tuple
    rect_width: float
    rect_length: float
    convex_hull: np.ndarray
    min_area_rect: tuple
    quality_score: float
    timestamp: float
    raw_polygon: np.ndarray
    smoothed_polygon: np.ndarray

class FootprintStabilizer:
    def __init__(self):
        self.history = {}

    def process(self, 
                vg, 
                mask: np.ndarray, 
                homography, 
                frame_w: int, 
                frame_h: int, 
                timestamp: float) -> Tuple[FootprintGeometry, Dict[str, Any]]:
        
        debug_geometry = {
            "mask": mask.copy(),
            "ground_points": [],
            "strip_candidates": [],
            "accepted_strips": [],
            "convex_hull": None,
            "min_area_rect": None,
            "pca_axis": None,
            "orientation_vector": None,
            "filtered_points": None,
            "quality": 0.0,
            "mask_completeness": 0.0,
            "ground_density": 0.0,
            "hull_consistency": 0.0,
            "temporal_stability": 0.0,
            "history_polygons": []
        }
        
        print(f"\nVehicle {vg.id}")
        
        # If vg was rejected, we skip
        if vg.validation_status != "Valid":
            print(f"Vehicle {vg.id} rejected at Input. Reason: Validation status is not Valid")
            return None, debug_geometry
            
        print("✓ Mask")

        sx = homography.calib_width / frame_w
        sy = homography.calib_height / frame_h

        # Phase 1: MULTI STRIP EXTRACTION
        y_indices, x_indices = np.where(mask > 128)
        if len(y_indices) == 0:
            print(f"Vehicle {vg.id} rejected at Ground Strip. Reason: Empty segmentation mask")
            return None, debug_geometry
            
        bottom_y = np.max(y_indices)
        top_y = np.min(y_indices)
        
        strips = []
        current_y = bottom_y
        for i in range(5):
            if current_y < top_y:
                break
            
            strip_mask = np.zeros_like(mask)
            y_start = max(top_y, current_y - 3)
            y_end = current_y + 1
            strip_mask[y_start:y_end, :] = mask[y_start:y_end, :]
            
            sy_ind, sx_ind = np.where(strip_mask > 128)
            if len(sx_ind) == 0:
                current_y -= 4
                continue
                
            left_x = np.min(sx_ind)
            right_x = np.max(sx_ind)
            center_x = int(np.mean(sx_ind))
            pixel_count = len(sx_ind)
            
            contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            sx_sorted = np.sort(sx_ind)
            gaps = np.diff(sx_sorted)
            if len(gaps) > 0:
                breaks = np.where(gaps > 1)[0]
                if len(breaks) == 0:
                    continuity = len(sx_sorted)
                else:
                    lengths = np.diff(np.concatenate(([-1], breaks, [len(sx_sorted)-1])))
                    continuity = np.max(lengths)
            else:
                continuity = 1
                
            density = pixel_count / (right_x - left_x + 1) if (right_x - left_x + 1) > 0 else 0
            
            strips.append({
                'id': i,
                'left': (left_x, current_y),
                'right': (right_x, current_y),
                'center': (center_x, current_y),
                'pixel_count': pixel_count,
                'components': len(contours),
                'continuity': continuity,
                'density': density,
                'y_start': y_start,
                'y_end': y_end
            })
            
            debug_geometry["strip_candidates"].append(strips[-1])
            current_y -= 4

        # Phase 2: STRIP QUALITY ANALYSIS
        valid_strips = []
        for s in strips:
            score = 100.0
            # Temporarily disable strong quality filters, keep only basic ones
            if s['pixel_count'] < 5: score -= 50
            if score > 0:
                s['quality'] = score
                valid_strips.append(s)
                debug_geometry["accepted_strips"].append(s)

        if not valid_strips:
            print(f"Vehicle {vg.id} rejected at Ground Strip. Reason: No valid strips found")
            return None, debug_geometry
            
        print("✓ Ground Strip")

        # Phase 3: ADAPTIVE STRIP FUSION
        total_weight = 0
        fus_l, fus_r, fus_c = np.zeros(2), np.zeros(2), np.zeros(2)
        
        for s in valid_strips:
            base_weight = 1.0 / (s['id'] + 1)
            weight = base_weight * s['quality'] * s['density']
            fus_l += np.array(s['left']) * weight
            fus_r += np.array(s['right']) * weight
            fus_c += np.array(s['center']) * weight
            total_weight += weight
            
        fus_l /= total_weight
        fus_r /= total_weight
        fus_c /= total_weight
        
        bev_left = homography.project_point(fus_l[0] * sx, fus_l[1] * sy)
        bev_right = homography.project_point(fus_r[0] * sx, fus_r[1] * sy)
        bev_center = homography.project_point(fus_c[0] * sx, fus_c[1] * sy)

        # Phase 4: GROUND POINT CLOUD
        cloud_pixels_bev = []
        for s in valid_strips:
            y_start, y_end = s['y_start'], s['y_end']
            strip_y, strip_x = np.where(mask[y_start:y_end, :] > 128)
            strip_y += y_start
            for px, py in zip(strip_x, strip_y):
                bpx, bpy = homography.project_point(px * sx, py * sy)
                if not math.isnan(bpx) and not math.isinf(bpx):
                    cloud_pixels_bev.append([bpx, bpy])
                    
        if len(cloud_pixels_bev) < 3:
            print(f"Vehicle {vg.id} rejected at Projection. Reason: Fewer than 3 projected points")
            return None, debug_geometry
            
        cloud_pts = np.array(cloud_pixels_bev, dtype=np.float32)
        debug_geometry["ground_points"] = cloud_pts

        # Phase 8: OUTLIER REMOVAL (Median Filtering)
        med_x, med_y = np.median(cloud_pts[:, 0]), np.median(cloud_pts[:, 1])
        std_x, std_y = np.std(cloud_pts[:, 0]), np.std(cloud_pts[:, 1])
        dist_x = np.abs(cloud_pts[:, 0] - med_x)
        dist_y = np.abs(cloud_pts[:, 1] - med_y)
        
        valid_mask = (dist_x < 2 * std_x) & (dist_y < 2 * std_y)
        filtered_pts = cloud_pts[valid_mask]
        
        if len(filtered_pts) < 3:
            filtered_pts = cloud_pts
        debug_geometry["filtered_points"] = filtered_pts

        print("✓ Projection")

        # Phase 5: CONVEX HULL
        if len(filtered_pts) < 3:
            print(f"Vehicle {vg.id} rejected at Convex Hull. Reason: fewer than 3 projected points after filtering")
            return None, debug_geometry
            
        hull = cv2.convexHull(filtered_pts)
        debug_geometry["convex_hull"] = hull
        print("✓ Hull")

        # Phase 6: MINIMUM AREA RECTANGLE
        rect = cv2.minAreaRect(filtered_pts)
        debug_geometry["min_area_rect"] = rect
        print("✓ Rectangle")

        # Phase 7: PCA ORIENTATION
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(filtered_pts, np.empty((0)))
        major_vec = eigenvectors[0]
        minor_vec = eigenvectors[1]
        
        if major_vec[1] > 0:
            major_vec = -major_vec
            
        orientation = (float(major_vec[0]), float(major_vec[1]))
        debug_geometry["pca_axis"] = {
            'mean': mean[0], 'major': major_vec, 'minor': minor_vec, 'eigenvalues': eigenvalues
        }
        debug_geometry["orientation_vector"] = orientation
        print("✓ PCA")

        # Phase 10: QUALITY & CONFIDENCE (Filters Disabled)
        mask_completeness = min(100.0, (len(valid_strips) / 5.0) * 100.0)
        
        avg_density = np.mean([s['density'] for s in valid_strips]) if valid_strips else 0.0
        ground_density = min(100.0, avg_density * 100.0)
        
        var_ratio = eigenvalues[0][0] / (eigenvalues[1][0] + 1e-6)
        hull_consistency = min(100.0, max(0.0, (var_ratio - 1.0) * 20.0))
        
        temporal_stability = 100.0
        if vg.id in self.history:
            hist_ori = self.history[vg.id]['ori']
            cos_sim = np.dot(major_vec, hist_ori) / (np.linalg.norm(major_vec) * np.linalg.norm(hist_ori) + 1e-6)
            temporal_stability = max(0.0, float(abs(cos_sim)) * 100.0)

        quality = 0.25 * mask_completeness + 0.25 * ground_density + 0.25 * hull_consistency + 0.25 * temporal_stability
        
        debug_geometry["mask_completeness"] = mask_completeness
        debug_geometry["ground_density"] = ground_density
        debug_geometry["hull_consistency"] = hull_consistency
        debug_geometry["temporal_stability"] = temporal_stability
        debug_geometry["quality"] = quality

        # Construct raw footprint polygon (using the box points of min area rect as stable representation)
        raw_box = cv2.boxPoints(rect)

        # Phase 9: TEMPORAL STABILIZATION (EMA)
        alpha = 0.3 
        smoothed_box = raw_box.copy()
        if vg.id in self.history:
            hist = self.history[vg.id]
            sm_l = alpha * np.array(bev_left) + (1 - alpha) * hist['l']
            sm_c = alpha * np.array(bev_center) + (1 - alpha) * hist['c']
            sm_r = alpha * np.array(bev_right) + (1 - alpha) * hist['r']
            
            hist_ori = np.array(hist['ori'])
            if np.dot(major_vec, hist_ori) < 0:
                major_vec = -major_vec
            sm_ori = alpha * major_vec + (1 - alpha) * hist_ori
            sm_ori = sm_ori / (np.linalg.norm(sm_ori) + 1e-6)
            
            # Smooth the box itself
            smoothed_box = alpha * raw_box + (1 - alpha) * hist['box']
            
            debug_geometry["history_polygons"].append(hist['box'].copy()) # Previous
            debug_geometry["history_polygons"].append(raw_box.copy())     # Current
            debug_geometry["history_polygons"].append(smoothed_box.copy())# Smoothed
            
            # Increment history length for summary
            hist_length = hist.get('length', 1) + 1
            
            self.history[vg.id] = {'l': sm_l, 'c': sm_c, 'r': sm_r, 'ori': sm_ori, 'box': smoothed_box, 'length': hist_length}
            bev_left, bev_center, bev_right, orientation = tuple(sm_l), tuple(sm_c), tuple(sm_r), tuple(sm_ori)
        else:
            self.history[vg.id] = {
                'l': np.array(bev_left), 'c': np.array(bev_center), 'r': np.array(bev_right), 
                'ori': np.array(orientation), 'box': raw_box, 'length': 1
            }
            hist_length = 1
            debug_geometry["history_polygons"].append(raw_box.copy())
            debug_geometry["history_polygons"].append(raw_box.copy())
            debug_geometry["history_polygons"].append(raw_box.copy())
            
        debug_geometry["history_length"] = hist_length
        print("✓ History")

        fg = FootprintGeometry(
            vehicle_id=vg.id,
            footprint_points=filtered_pts,
            left_point=bev_left,
            center_point=bev_center,
            right_point=bev_right,
            orientation_vector=orientation,
            rect_width=min(rect[1][0], rect[1][1]),
            rect_length=max(rect[1][0], rect[1][1]),
            convex_hull=hull,
            min_area_rect=rect,
            quality_score=quality,
            timestamp=timestamp,
            raw_polygon=raw_box,
            smoothed_polygon=smoothed_box
        )

        return fg, debug_geometry
