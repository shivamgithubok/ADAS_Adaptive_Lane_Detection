import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DebugFlags:
    SHOW_ROI:          bool = True
    SHOW_BINARY_MASK:  bool = True
    SHOW_LANE_LINES:   bool = True
    SHOW_VEHICLES:     bool = True

_C = {
    'left':       (0,   80,  255),
    'right':      (255, 60,  0),
    'center':     (0,   255, 255),
    'roi':        (0,   220, 255),
    'ok':         (0,   255, 80),
    'warn':       (0,   140, 255),
    'err':        (0,   0,   255),
    'text':       (220, 220, 220),
    'header':     (0,   220, 255),
}

# ── Render-mode colour palette ────────────────────────────────────────────────
_RENDER_COLORS = {
    'FULL':      (0,   255, 80),   # bright green
    'PARTIAL':   (0,   200, 255),  # amber
    'OUTLINE':   (180, 180, 0),    # teal
    'POINT':     (140, 140, 140),  # gray
    'PREDICTED': (0,   100, 255),  # orange-red
}

def _put(img, text, pos, scale=0.45, color=None, thickness=1, bold=False):
    color = color or _C['text']
    if bold:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _hline(img, y, color=(60, 60, 60)):
    cv2.line(img, (0, y), (img.shape[1], y), color, 1)

def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8, gap_len=5):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        dx = ex - sx
        dy = ey - sy
        length = int(np.hypot(dx, dy))
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        drawn = 0
        while drawn < length:
            seg_end = min(drawn + dash_len, length)
            cv2.line(img,
                     (int(sx + ux * drawn), int(sy + uy * drawn)),
                     (int(sx + ux * seg_end), int(sy + uy * seg_end)),
                     color, thickness)
            drawn = seg_end + gap_len

class LaneVisualizer:
    def __init__(self, flags: Optional[DebugFlags] = None):
        self.bev_h = 800
        self.bev_w = 800
        self.panel_h = 800
        self.panel_w = 500
        self.flags = flags or DebugFlags()

    # ══════════════════════════════════════════════════════════════════════
    # Window 1 — Camera View with Adaptive Vehicle Rendering
    # ══════════════════════════════════════════════════════════════════════

    def draw_window1(self, frame, bev_road, lane_debug, vehicles, odz_states, flags=None, occlusion_states=None):
        flags = flags or self.flags
        out = frame.copy()
        
        # ── ROI ─────────────────────────────────────────────────────
        if flags.SHOW_ROI and lane_debug.get('roi_poly') is not None:
            roi_pts = lane_debug['roi_poly']
            cv2.polylines(out, [roi_pts], True, _C['roi'], 2)
            
        # ── Binary Mask / HLS ─────────────────────────────────────────────
        if flags.SHOW_BINARY_MASK and lane_debug.get('clean_mask') is not None:
            bm = lane_debug['clean_mask']
            white_overlay = np.zeros_like(out)
            white_overlay[bm > 0] = (255, 255, 255)
            cv2.addWeighted(white_overlay, 0.35, out, 1.0, 0, out)

        # ── Image Space Lines ───────────────────────────────────────
        if flags.SHOW_LANE_LINES:
            ld_result = lane_debug.get('ld_result')
            if ld_result is not None:
                if ld_result.sampled_image_points_left is not None:
                    pts = np.array(ld_result.sampled_image_points_left, np.int32)
                    cv2.polylines(out, [pts], False, _C['left'], 3)
                    for pt in pts[::10]:
                        cv2.circle(out, tuple(pt), 2, (255, 255, 255), -1)
                if ld_result.sampled_image_points_right is not None:
                    pts = np.array(ld_result.sampled_image_points_right, np.int32)
                    cv2.polylines(out, [pts], False, _C['right'], 3)
                    for pt in pts[::10]:
                        cv2.circle(out, tuple(pt), 2, (255, 255, 255), -1)
            else:
                left_line = lane_debug.get('left_line')
                if left_line:
                    cv2.line(out, (left_line[0], left_line[1]), (left_line[2], left_line[3]), _C['left'], 3)
                right_line = lane_debug.get('right_line')
                if right_line:
                    cv2.line(out, (right_line[0], right_line[1]), (right_line[2], right_line[3]), _C['right'], 3)
                
        # ── Vehicle rendering (occlusion-aware) ──────────────────────
        if flags.SHOW_VEHICLES:
            state_map = {s.vehicle_id: s for s in odz_states}

            # Build occlusion lookup
            occ_map = {}
            if occlusion_states:
                occ_map = {s.vehicle_id: s for s in occlusion_states}

            # Sort vehicles by render_priority (furthest first = drawn first,
            # nearest last = drawn on top)
            sorted_vehicles = list(vehicles)
            if occ_map:
                sorted_vehicles.sort(
                    key=lambda v: occ_map.get(v.get('track_id', -1),
                                              type('', (), {'render_priority': 999})).render_priority,
                    reverse=True  # highest priority number (far) drawn first
                )

            occupied_labels = []

            for v in sorted_vehicles:
                track_id = v.get('track_id', -1)
                state = state_map.get(track_id)
                occ = occ_map.get(track_id)

                if not state:
                    continue

                # Determine render mode from occlusion state, default to FULL
                render_mode = occ.render_mode if occ else "FULL"
                color_bgr = _RENDER_COLORS.get(render_mode, (0, 255, 0))

                # Override color for ODZ-INACTIVE vehicles (non-occlusion reasons)
                if state.status == "INACTIVE":
                    color_bgr = (0, 0, 255) if state.reason == "Image Border" else (128, 128, 128)
                    # INACTIVE vehicles always render as OUTLINE regardless of occlusion
                    render_mode = "OUTLINE"

                bbox = v.get('bbox', [0, 0, 0, 0])
                cx = int((bbox[0] + bbox[2]) // 2)
                cy = int(bbox[1] - 10)

                dist = 0.0
                dist_str = ""
                if state.geometry and state.geometry.metric_estimate:
                    dist = state.geometry.metric_estimate.distance_m
                    dist_str = f" | {dist:.1f}m"

                # ── Adaptive rendering per mode ────────────────────
                if render_mode == "FULL":
                    # Full bounding box + mask overlay + class label + distance
                    if 'mask_img' in v:
                        mask = v['mask_img']
                        color_mask = np.zeros_like(out)
                        color_mask[mask > 128] = color_bgr
                        out = cv2.addWeighted(out, 1.0, color_mask, 0.3, 0)

                    cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)

                    text = f"V{track_id}" if dist > 40.0 else f"V{track_id}{dist_str}"
                    font_scale = 0.6 if dist < 20 else 0.4

                elif render_mode == "PARTIAL":
                    # Full bounding box (no mask overlay) + smaller label
                    cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)

                    text = f"V{track_id}{dist_str}" if dist <= 40.0 else f"V{track_id}"
                    font_scale = 0.4

                elif render_mode == "OUTLINE":
                    # Thin outline only + distance
                    cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 1)

                    text = f"{dist:.0f}m" if dist > 0 else f"V{track_id}"
                    font_scale = 0.35

                elif render_mode == "POINT":
                    # Center point + track ID only
                    bcx = int((bbox[0] + bbox[2]) // 2)
                    bcy = int((bbox[1] + bbox[3]) // 2)
                    cv2.circle(out, (bcx, bcy), 4, color_bgr, -1)
                    cv2.circle(out, (bcx, bcy), 6, color_bgr, 1)

                    text = f"V{track_id}"
                    font_scale = 0.3
                    cy = bcy - 12

                elif render_mode == "PREDICTED":
                    # Dashed rectangle + prediction marker
                    _draw_dashed_rect(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 1)
                    bcx = int((bbox[0] + bbox[2]) // 2)
                    bcy = int((bbox[1] + bbox[3]) // 2)
                    # Draw crosshair
                    cv2.line(out, (bcx - 8, bcy), (bcx + 8, bcy), color_bgr, 1)
                    cv2.line(out, (bcx, bcy - 8), (bcx, bcy + 8), color_bgr, 1)

                    text = f"?V{track_id}"
                    font_scale = 0.3
                    cy = bcy - 12
                else:
                    text = f"V{track_id}"
                    font_scale = 0.4

                # ── Anti-collision label placement ─────────────────
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                tw, th = text_size

                placed = False
                for ox, oy in [(0, 0), (0, -25), (0, 25), (-tw - 10, 0), (tw + 10, 0), (0, -45)]:
                    nx, ny = cx - 20 + ox, cy + oy
                    collision = any(
                        nx < bx + bw and nx + tw > bx and ny - th < by and ny > by - bh
                        for bx, by, bw, bh in occupied_labels
                    )
                    if not collision:
                        occupied_labels.append((nx, ny, tw, th))
                        if ox != 0 or oy != 0:
                            cv2.line(out, (cx, cy), (nx + tw // 2, ny), color_bgr, 1)
                        cv2.putText(out, text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2)
                        placed = True
                        break

                if not placed:
                    cv2.putText(out, text, (cx - 20, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2)
                    occupied_labels.append((cx - 20, cy - 40, tw, th))

        return out

    # ══════════════════════════════════════════════════════════════════════
    # Window 2 — BEV Footprint View (occlusion-aware)
    # ══════════════════════════════════════════════════════════════════════

    def draw_window2(self, bev_road, odz_states: List, homography=None, occlusion_states=None) -> np.ndarray:
        canvas = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)

        PPM = homography.PPM if homography else 145.0 / 3.70

        # Build occlusion lookup
        occ_map = {}
        if occlusion_states:
            occ_map = {s.vehicle_id: s for s in occlusion_states}

        # Metric grid
        for y in range(800, 0, -int(10 * PPM)):
            cv2.line(canvas, (0, y), (800, y), (50, 50, 50), 1)
        for x in range(0, 800, int(3.7 * PPM)):
            cv2.line(canvas, (x, 0), (x, 800), (50, 50, 50), 1)

        # ODZ overlay
        odz_max_y   = int(800 - 40.0 * PPM)
        odz_min_x   = int(400 - 5.55 * PPM)
        odz_max_x   = int(400 + 5.55 * PPM)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (odz_min_x, odz_max_y), (odz_max_x, 800), (30, 30, 0), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Ego vehicle
        ego_w_px = int(0.5 * PPM)
        ego_l_px = int(1.0 * PPM)
        cv2.rectangle(canvas,
                      (400 - ego_w_px // 2, 800 - ego_l_px),
                      (400 + ego_w_px // 2, 800),
                      (255, 0, 0), -1)
        cv2.putText(canvas, "EGO", (400 - ego_w_px // 2, 800 - ego_l_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Lane curves
        if bev_road:
            lane_pairs = [
                (bev_road.left_lane,   _C['left'],   'L'),
                (bev_road.right_lane,  _C['right'],  'R'),
                (bev_road.centerline,  _C['center'], 'C'),
            ]
            for curve, color, label in lane_pairs:
                if curve and curve.sampled_points is not None and len(curve.sampled_points) > 1:
                    # Draw original projected points first
                    if getattr(curve, 'bev_points', None) is not None:
                        for pt in curve.bev_points:
                            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (200, 200, 200), -1)
                    
                    pts = np.array(curve.sampled_points, dtype=np.int32)
                    thickness = 2 if label != 'C' else 1
                    cv2.polylines(canvas, [pts], False, color, thickness)
                    tp = pts[-1] if len(pts) > 0 and pts[-1][1] < pts[0][1] else pts[0]
                    if 0 <= tp[0] < 800 and 0 <= tp[1] < 800:
                        _put(canvas, label, (tp[0] + 4, tp[1]), scale=0.5, color=color)
            
            # Draw 5m width markers
            if bev_road.left_lane and bev_road.right_lane:
                for y_m in range(5, 50, 5):
                    y_px = 800 - int(y_m * PPM)
                    if y_px < 0:
                        break
                    cl = bev_road.left_lane.bev_polynomial
                    cr = bev_road.right_lane.bev_polynomial
                    xl = int(cl[0]*(y_px**2) + cl[1]*y_px + cl[2])
                    xr = int(cr[0]*(y_px**2) + cr[1]*y_px + cr[2])
                    cv2.line(canvas, (xl, y_px), (xr, y_px), (100, 100, 100), 1, cv2.LINE_AA)
                    _put(canvas, f"{y_m}m", ((xl+xr)//2 - 15, y_px - 5), scale=0.4, color=(150, 150, 150))

        # ── Footprints (occlusion-aware) ──────────────────────────────
        # Draw INACTIVE footprints first (background layer)
        for state in [s for s in odz_states if s.status == "INACTIVE"]:
            vme = state.geometry.metric_estimate
            if vme and vme.corrected_polygon is not None:
                cv2.fillPoly(canvas, [np.array(vme.corrected_polygon, dtype=np.int32)], (70, 70, 70))

        # Draw ACTIVE footprints with occlusion-aware styling
        occupied_bev_labels = []
        for state in [s for s in odz_states if s.status == "ACTIVE"]:
            vme = state.geometry.metric_estimate
            if vme and vme.corrected_polygon is not None:
                pts = np.array(vme.corrected_polygon, dtype=np.int32)
                occ = occ_map.get(state.vehicle_id)
                render_mode = occ.render_mode if occ else "FULL"

                # Choose footprint style based on occlusion state
                if render_mode in ("FULL", "PARTIAL"):
                    fp_color = (0, 255, 0)
                    fp_thickness = 2
                    cv2.polylines(canvas, [pts], True, fp_color, fp_thickness)
                elif render_mode == "OUTLINE":
                    fp_color = (0, 180, 180)
                    cv2.polylines(canvas, [pts], True, fp_color, 1)
                elif render_mode == "POINT":
                    fp_color = (100, 100, 100)
                    cx_fp = int(np.mean(pts[:, 0]))
                    cy_fp = int(np.mean(pts[:, 1]))
                    cv2.circle(canvas, (cx_fp, cy_fp), 5, fp_color, -1)
                elif render_mode == "PREDICTED":
                    fp_color = (0, 100, 255)
                    cx_fp = int(np.mean(pts[:, 0]))
                    cy_fp = int(np.mean(pts[:, 1]))
                    cv2.circle(canvas, (cx_fp, cy_fp), 6, fp_color, 2)  # hollow circle
                    cv2.line(canvas, (cx_fp - 5, cy_fp), (cx_fp + 5, cy_fp), fp_color, 1)
                    cv2.line(canvas, (cx_fp, cy_fp - 5), (cx_fp, cy_fp + 5), fp_color, 1)
                else:
                    fp_color = (0, 255, 0)
                    cv2.polylines(canvas, [pts], True, fp_color, 2)

                # Heading arrow (only for FULL/PARTIAL)
                if render_mode in ("FULL", "PARTIAL"):
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    front_mid = (pts[0] + pts[1]) / 2.0
                    cv2.arrowedLine(canvas, (cx, cy), (int(front_mid[0]), int(front_mid[1])), fp_color, 2)

                # Label
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                text = f"V{vme.vehicle_id}"
                tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                placed = False
                for ox, oy in [(10, 0), (10, -20), (10, 20), (-tw - 10, 0), (0, -25)]:
                    nx, ny = cx + ox, cy + oy
                    if not any(nx < bx + bw and nx + tw > bx and ny - th < by and ny > by - bh
                               for bx, by, bw, bh in occupied_bev_labels):
                        occupied_bev_labels.append((nx, ny, tw, th))
                        label_color = fp_color if render_mode not in ("FULL", "PARTIAL") else (255, 255, 255)
                        _put(canvas, text, (nx, ny), scale=0.5, color=label_color)
                        placed = True
                        break
                if not placed:
                    _put(canvas, text, (cx + 10, cy), scale=0.5, color=(255, 255, 255))

        # Crop the sides to make the window narrower (500x800 instead of 800x800)
        canvas_cropped = canvas[:, 150:650]
        return canvas_cropped

    # ══════════════════════════════════════════════════════════════════════
    # Window 3 — Debug Panel (with occlusion statistics)
    # ══════════════════════════════════════════════════════════════════════

    def draw_window3(self, lane_debug, road_geom, occlusion_states=None) -> np.ndarray:
        canvas = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        y = 0

        def section(title):
            nonlocal y
            y += 6
            _hline(canvas, y, (80, 80, 80))
            y += 18
            _put(canvas, title, (10, y), scale=0.52, color=_C['header'], bold=True)
            y += 22

        def row(label, value, ok=None):
            nonlocal y
            color = _C['ok'] if ok is True else _C['err'] if ok is False else _C['text']
            _put(canvas, f"{label:<22} {value}", (14, y), scale=0.44, color=color)
            y += 18

        # ── Header ─────────────────────────────────────────────────────────────
        y = 24
        _put(canvas, "ADAS LANE DEBUG PANEL", (10, y), scale=0.6, color=_C['header'], bold=True)

        # ── Stage 1: Detector output ───────────────────────────────────────────────────────
        section("LANE DETECTION (v0 API)")
        row("Calibration", "LOCKED" if lane_debug.get('calibrated') else "WARMUP", ok=lane_debug.get('calibrated'))
        car_boxes = lane_debug.get('car_boxes', [])
        row("Detected Cars (YOLO)", str(len(car_boxes)))
        
        ld_result = lane_debug.get('ld_result')
        if ld_result:
            row("Lane Pixels", str(ld_result.pixel_count))
            left_samples = len(ld_result.sampled_image_points_left) if ld_result.sampled_image_points_left is not None else 0
            right_samples = len(ld_result.sampled_image_points_right) if ld_result.sampled_image_points_right is not None else 0
            row("Image Sample Count", f"L:{left_samples} | R:{right_samples}")
            row("Image Polynomial Left", "FITTED" if ld_result.left_polynomial is not None else "LOST", ok=ld_result.left_polynomial is not None)
            row("Image Polynomial Right", "FITTED" if ld_result.right_polynomial is not None else "LOST", ok=ld_result.right_polynomial is not None)
            row("Detection Confidence", f"{ld_result.confidence:.1f}%", ok=ld_result.confidence > 60)
        else:
            left = lane_debug.get('left_line')
            right = lane_debug.get('right_line')
            row("Left Line", "DETECTED" if left else "LOST", ok=left is not None)
            row("Right Line", "DETECTED" if right else "LOST", ok=right is not None)

        # ── Stage 2: BEV Metrics ───────────────────────────────────────────────
        section("BEV METRICS")
        if road_geom:
            lw = road_geom.lane_width
            row("Lane Width", f"{lw:.2f} m" if lw > 0 else "N/A", ok=(2.5 < lw < 5.5) if lw > 0 else False)
            
            if road_geom.left_lane:
                row("Left Model Type", road_geom.left_lane.model_type)
                row("Left Curvature (R)", f"{road_geom.left_lane.curvature_m:.0f}m")
                pts_len = len(road_geom.left_lane.bev_points) if getattr(road_geom.left_lane, 'bev_points', None) is not None else 0
                row("Left BEV Points", str(pts_len))
            if road_geom.right_lane:
                row("Right Model Type", road_geom.right_lane.model_type)
                row("Right Curvature (R)", f"{road_geom.right_lane.curvature_m:.0f}m")
                pts_len = len(road_geom.right_lane.bev_points) if getattr(road_geom.right_lane, 'bev_points', None) is not None else 0
                row("Right BEV Points", str(pts_len))

        # ── Stage 3: Occlusion Statistics ──────────────────────────────────────
        if occlusion_states is not None:
            section("OCCLUSION MANAGEMENT")
            total = len(occlusion_states)
            row("Tracked Vehicles", str(total))

            if total > 0:
                n_full = sum(1 for s in occlusion_states if s.render_mode == "FULL")
                n_partial = sum(1 for s in occlusion_states if s.render_mode == "PARTIAL")
                n_outline = sum(1 for s in occlusion_states if s.render_mode == "OUTLINE")
                n_point = sum(1 for s in occlusion_states if s.render_mode == "POINT")
                n_predicted = sum(1 for s in occlusion_states if s.render_mode == "PREDICTED")

                row("Fully Visible", str(n_full), ok=True)
                if n_partial > 0:
                    row("Partially Occluded", str(n_partial), ok=True)
                if n_outline > 0:
                    row("Heavily Occluded", str(n_outline), ok=None)
                if n_point > 0:
                    row("Minimal Visible", str(n_point), ok=None)
                if n_predicted > 0:
                    row("Predicted Only", str(n_predicted), ok=False)

                # Per-vehicle detail for occluded vehicles
                occluded = [s for s in occlusion_states if s.render_mode not in ("FULL",)]
                if occluded:
                    y += 4
                    for s in occluded[:5]:  # Cap at 5 to avoid overflow
                        vis_str = f"V{s.vehicle_id}: vis={s.visibility_score:.0f}% occ={s.occlusion_score:.0f}% [{s.render_mode}]"
                        color = _C['warn'] if s.visibility_score > 20 else _C['err']
                        _put(canvas, vis_str, (18, y), scale=0.38, color=color)
                        y += 16

        return canvas
