import cv2
import numpy as np
import math
import config

class Visualizer:
    @staticmethod
    def draw(frame, obj_data, seg_data, contact_data, track_info, debug_mode=0):
        draw_bbox = debug_mode in (1, 6)
        draw_seg = debug_mode in (2, 6)
        draw_contact = debug_mode in (3, 6)
        draw_tracking = debug_mode in (4, 6)
        draw_clean = debug_mode == 0
        
        x1, y1, x2, y2 = obj_data['bbox']
        track_id = obj_data.get('track_id', '?')
        score = obj_data.get('score', 0)
        
        # 1. Segmentation Mask
        if draw_seg or draw_clean:
            if 'mask_img' in obj_data and obj_data['mask_img'] is not None:
                mask = obj_data['mask_img']
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = config.COLOR_MASK
                mask_indices = mask > 0
                frame[mask_indices] = cv2.addWeighted(
                    frame[mask_indices], 1 - config.ALPHA_MASK,
                    colored_mask[mask_indices], config.ALPHA_MASK, 0
                )
            # Outline
            if seg_data and 'contour' in seg_data and len(seg_data['contour']) > 0:
                contour_pts = np.array(seg_data['contour'], dtype=np.int32)
                cv2.drawContours(frame, [contour_pts], -1, config.COLOR_MASK, 2)

        # 2. Bounding Box
        if draw_bbox or draw_clean:
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_BBOX, config.THICKNESS_BBOX)

        # 3. Adaptive Strip
        if draw_contact or (debug_mode != 0 and getattr(config, 'DEBUG_GROUND_CONTACT', True)):
            if contact_data is not None:
                sx1, sy1, sx2, sy2 = contact_data['strip_bbox']
                
                if getattr(config, 'DRAW_STRIP', True):
                    strip_overlay = frame.copy()
                    cv2.rectangle(strip_overlay, (sx1, sy1), (sx2, sy2), config.COLOR_STRIP, -1)
                    cv2.addWeighted(strip_overlay, getattr(config, 'ALPHA_STRIP', 0.3), frame, 1 - getattr(config, 'ALPHA_STRIP', 0.3), 0, frame)
                    # draw border
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), config.COLOR_STRIP, config.THICKNESS_STRIP)

        # 4. Ground Contact Line & Points
        if draw_contact or (debug_mode != 0 and getattr(config, 'DEBUG_GROUND_CONTACT', True)):
            if contact_data is not None and getattr(config, 'DRAW_CONTACT_POINTS', True):
                left_pt = contact_data.get('left_contact', (np.nan, np.nan))
                right_pt = contact_data.get('right_contact', (np.nan, np.nan))
                
                valid_left = not math.isnan(left_pt[0]) and not math.isnan(left_pt[1])
                valid_right = not math.isnan(right_pt[0]) and not math.isnan(right_pt[1])
                
                # Ground contact line
                if valid_left and valid_right:
                    cv2.line(frame, left_pt, right_pt, getattr(config, 'COLOR_GROUND_LINE', (0,255,255)), 2)
                
                # Ground contact points
                if valid_left:
                    cv2.circle(frame, left_pt, config.RADIUS_CONTACT, config.COLOR_L_CONTACT, -1)
                if valid_right:
                    cv2.circle(frame, right_pt, config.RADIUS_CONTACT, config.COLOR_R_CONTACT, -1)
                    
                # Center point
                median_pt = contact_data.get('median_contact', (np.nan, np.nan))
                if not math.isnan(median_pt[0]) and not math.isnan(median_pt[1]):
                    cv2.circle(frame, median_pt, config.RADIUS_CONTACT, config.COLOR_M_CONTACT, -1)

        # 5. Label
        if draw_clean or debug_mode != 0:
            if score >= 90:
                score_color = (0, 255, 0)
            elif score >= 70:
                score_color = (0, 255, 255)
            else:
                score_color = (0, 0, 255)

            y_offset = y1 - 30
            if y_offset < 10: y_offset = y2 + 20
            
            # Clean ID + Distance (placeholder) + Score
            cv2.putText(frame, f"ID:{track_id}", (x1, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_LABEL, config.COLOR_TEXT, config.THICKNESS_TEXT)
            cv2.putText(frame, f"Q:{score}", (x1, y_offset + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_LABEL, score_color, config.THICKNESS_TEXT)

        # 6. Strip Statistics (Debug Mode Only)
        if debug_mode != 0 and contact_data is not None:
            stats_x = x2 + 5
            stats_y = y1 + 15
            
            strip_h = contact_data.get('strip_height', 0)
            ground_w = contact_data.get('ground_width', 0)
            ground_px = contact_data.get('ground_pixels', 0)
            
            # Draw next to bbox
            cv2.putText(frame, f"Strip = {strip_h} px", (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            cv2.putText(frame, f"Width = {ground_w} px", (stats_x, stats_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            cv2.putText(frame, f"Pixels = {ground_px}", (stats_x, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            cv2.putText(frame, f"Q = {score}", (stats_x, stats_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        return frame
        
    @staticmethod
    def draw_fps(frame, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_FPS, (255, 255, 255), 2)
        return frame

    @staticmethod
    def draw_debug_overlay(frame, stats_history, obj_count):
        # Draw small panel with overall stats
        panel_x, panel_y = 20, 60
        
        avg_strip = sum(stats_history['strip_heights']) / len(stats_history['strip_heights']) if stats_history['strip_heights'] else 0
        avg_gw = sum(stats_history['ground_widths']) / len(stats_history['ground_widths']) if stats_history['ground_widths'] else 0
        avg_score = sum(stats_history['scores']) / len(stats_history['scores']) if stats_history['scores'] else 0
        
        lines = [
            f"Objects: {obj_count}",
            f"Average Strip Height: {avg_strip:.0f} px",
            f"Average Ground Width: {avg_gw:.0f} px",
            f"Average Quality: {avg_score:.0f}",
            f"Failures: {stats_history['failed']}",
            f"Low Confidence: {stats_history['low_conf']}"
        ]
        
        # Background
        cv2.rectangle(frame, (panel_x - 5, panel_y - 15), (panel_x + 230, panel_y + len(lines) * 20 + 5), (0,0,0), -1)
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (panel_x, panel_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        return frame
