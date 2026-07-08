import cv2
import numpy as np
import math
class BEVVisualizer:
    def __init__(self, bev_size=(800, 800), ppm=145.0/3.70):
        self.width, self.height = bev_size
        self.scale = ppm
        self.origin_x = 400
        self.origin_y = 800

    def _meters_to_pixels(self, x, z):
        px = int(self.origin_x + x * self.scale)
        py = int(self.origin_y - z * self.scale)
        return (px, py)

    def draw_grid(self, frame):
        for z in range(0, 200, 5):
            py = int(self.origin_y - z * self.scale)
            if py < 0: continue
            cv2.line(frame, (0, py), (self.width, py), (40, 40, 40), 1)
            cv2.putText(frame, f"{z}m", (10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        for x in range(-50, 51, 5):
            px = int(self.origin_x + x * self.scale)
            cv2.line(frame, (px, 0), (px, self.height), (40, 40, 40), 1)

    def render(self, vehicles, frame_w=None, frame_h=None, calib_w=None, calib_h=None):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.draw_grid(frame)

        if frame_w and frame_h and calib_w and calib_h:
            sx = calib_w / frame_w
            sy = calib_h / frame_h
            cv2.putText(frame, f"Frame: {frame_w}x{frame_h}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Calibration: {calib_w}x{calib_h}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Scale Factor: X={sx:.3f}, Y={sy:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for vg in vehicles:
            if vg.validation_status == "Valid":
                color = (0, 255, 0)
            elif vg.validation_status == "Warning":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            # Draw footprint
            pts = [
                self._meters_to_pixels(*vg.footprint_fl),
                self._meters_to_pixels(*vg.footprint_fr),
                self._meters_to_pixels(*vg.footprint_rr),
                self._meters_to_pixels(*vg.footprint_rl)
            ]
            poly = np.array([pts], dtype=np.int32)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            # Draw center point
            center_px = self._meters_to_pixels(*vg.bev_center)
            cv2.circle(frame, center_px, 3, color, -1)

            # Draw left/right contact points (already in pixels)
            l_valid = not (math.isnan(vg.bev_left[0]) or math.isinf(vg.bev_left[0]))
            r_valid = not (math.isnan(vg.bev_right[0]) or math.isinf(vg.bev_right[0]))
            c_valid = not (math.isnan(vg.bev_center[0]) or math.isinf(vg.bev_center[0]))
            
            if l_valid and r_valid:
                l_px = (int(vg.bev_left[0]), int(vg.bev_left[1]))
                r_px = (int(vg.bev_right[0]), int(vg.bev_right[1]))
                cv2.circle(frame, l_px, 3, (0, 0, 255), -1)
                cv2.circle(frame, r_px, 3, (255, 0, 0), -1)
                
                # Step 7: Draw one horizontal line between the projected left and right points
                cv2.line(frame, l_px, r_px, color, 1)

            # Draw Text
            label = f"ID:{vg.id}"
            w_label = f"W:{vg.width_m:.1f}m"
            c_label = f"({vg.bev_center[0]:.1f}, {vg.bev_center[1]:.1f})"
            
            text_x, text_y = pts[0]
            if not l_valid or not r_valid:
                text_x, text_y = 400, 400 # Default to center if projection failed
                
            cv2.putText(frame, label, (text_x, text_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, w_label, (text_x, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, c_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

class StabilizationVisualizer:
    def __init__(self, bev_size=(800, 800), panel_size=(400, 800)):
        self.width, self.height = bev_size
        self.panel_w, self.panel_h = panel_size
        
        self.colors = [
            (0, 255, 0),     # Green
            (255, 255, 0),   # Cyan
            (0, 255, 255),   # Yellow
            (0, 165, 255),   # Orange
            (255, 0, 255),   # Magenta
            (0, 0, 255),     # Red
            (255, 255, 255)  # White
        ]

    def render(self, vehicles_debug_list):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        
        if not vehicles_debug_list:
            return canvas, panel
            
        y_off = 30
        def draw_text(text, color=(255, 255, 255)):
            nonlocal y_off
            cv2.putText(panel, text, (15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_off += 25
            
        for fg, debug_data, vme, dim_est in vehicles_debug_list:
            if fg is None or debug_data is None:
                continue
                
            color = self.colors[fg.vehicle_id % len(self.colors)]
                
            # ==========================================
            # WINDOW 2: BEV VISUALIZATION
            # ==========================================
            
            # Draw Projected BEV cloud (cyan)
            if debug_data["filtered_points"] is not None:
                for pt in debug_data["filtered_points"]:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(canvas, (px, py), 2, (255, 255, 0), -1)
                    
            # Draw Convex Hull (yellow polygon)
            if debug_data["convex_hull"] is not None:
                hull = debug_data["convex_hull"].astype(np.int32)
                cv2.polylines(canvas, [hull], True, (0, 255, 255), 1)
                
            # Draw Original observed footprint (Thin Gray)
            if debug_data["min_area_rect"] is not None:
                box = cv2.boxPoints(debug_data["min_area_rect"])
                box = np.int32(box)
                cv2.drawContours(canvas, [box], 0, (100, 100, 100), 1)
                
            # Draw Final stabilized footprint (Phase 2B.1)
            if len(debug_data["history_polygons"]) >= 3:
                smooth_poly = np.int32(debug_data["history_polygons"][2])
                cv2.polylines(canvas, [smooth_poly], True, color, 1)

            # Draw Metric Corrected Footprint (Phase 2C.1.5) - (Thick Green)
            if vme is not None and vme.corrected_polygon is not None:
                comp_poly = np.int32(vme.corrected_polygon)
                cv2.polylines(canvas, [comp_poly], True, (0, 255, 0), 2)
                
                # Distance and Dimensions label on BEV canvas
                min_y = int(np.min(comp_poly[:, 1]))
                max_x = int(np.max(comp_poly[:, 0]))
                
                # Calculate correction % for display
                corr_pct = (vme.corrected_width_m * vme.corrected_length_m) / (vme.observed_width_m * vme.observed_length_m + 0.001) * 100.0 - 100.0
                
                cv2.putText(canvas, f"{vme.distance_m:.1f}m", (max_x + 10, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(canvas, f"{vme.corrected_width_m:.2f}x{vme.corrected_length_m:.2f}m", (max_x + 10, min_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(canvas, f"{corr_pct:+.1f}%", (max_x + 10, min_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
            # Draw PCA Axis (magenta)
            if debug_data["pca_axis"] is not None:
                mean = debug_data["pca_axis"]["mean"]
                major = fg.orientation_vector
                minor = debug_data["pca_axis"]["minor"]
                
                mx, my = int(mean[0]), int(mean[1])
                # Minor axis
                minor_len = 20
                px2 = int(mx + minor[0] * minor_len)
                py2 = int(my + minor[1] * minor_len)
                cv2.line(canvas, (mx, my), (px2, py2), (255, 0, 255), 1)
                
                # Major axis
                major_len = 40
                p_end = (int(mx + major[0] * major_len), int(my + major[1] * major_len))
                cv2.arrowedLine(canvas, (mx, my), p_end, (255, 0, 255), 2, tipLength=0.3)
                    
            # Center point (white)
            cv2.circle(canvas, (int(fg.center_point[0]), int(fg.center_point[1])), 4, (255, 255, 255), -1)
                
            # Vehicle ID (above footprint)
            if fg.footprint_points is not None and len(fg.footprint_points) > 0:
                min_y = int(np.min(fg.footprint_points[:, 1]))
                center_x = int(np.mean(fg.footprint_points[:, 0]))
                cv2.putText(canvas, f"ID:{fg.vehicle_id}", (center_x - 20, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ==========================================
            # WINDOW 3: GEOMETRY DEBUG PANEL (Summary)
            # ==========================================
            draw_text(f"--- ID {fg.vehicle_id} ---", color)
            
            if vme:
                draw_text(f"Source: Geometry Only", (255, 255, 255))
                draw_text(f"Distance: {vme.distance_m:.1f} m")
                heading = debug_data.get("heading", 0.0)
                draw_text(f"Heading: {heading:.1f} deg")
                draw_text(f"Ground Width: {vme.observed_width_m:.2f} m")
                
                ground_pts = len(debug_data.get("filtered_points", [])) if debug_data.get("filtered_points") is not None else 0
                draw_text(f"Ground Pixels: {ground_pts}")
                
                # Show Geometry Footprint only
                draw_text(f"Estimated Footprint Width: {vme.observed_width_m:.2f} m", (0, 255, 0))
                draw_text(f"Estimated Footprint Length: {vme.observed_length_m:.2f} m", (0, 255, 0))
                
                conf = vme.correction_confidence
                draw_text(f"Projection Confidence: {conf:.1f}%")
                
                temporal = debug_data.get("temporal_stability", 0.0) * 100
                draw_text(f"Temporal Stability: {temporal:.1f}%")
                
                hull = debug_data.get("hull_consistency", 0.0) * 100
                draw_text(f"Heading Stability: {hull:.1f}%")
                
                qual = debug_data.get("quality", 0.0) * 100
                draw_text(f"Geometry Confidence: {qual:.1f}%")
                
            y_off += 5
            

        return canvas, panel
