import cv2
import numpy as np
from collections import deque
import threading
import queue
import time

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import RANSACRegressor
    _RANSAC_AVAILABLE = True
except ImportError:
    _RANSAC_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO disabled. pip install ultralytics")

import sys
import os

_DEPTH_AVAILABLE = False
_DEPTH_MODEL = None
try:
    if os.path.exists('./depth_anything_v2_repo'):
        sys.path.append('./depth_anything_v2_repo')
        from depth_anything_v2.dpt import DepthAnythingV2
        _DEPTH_AVAILABLE = True
except ImportError:
    pass

def _init_depth_model():
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None: return
    print("=========================\nDepth Model Status\n")
    if not _DEPTH_AVAILABLE or not _TORCH_AVAILABLE:
        print("Model Loaded : NO\n=========================\n")
        return
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location='cpu'))
        model.eval().to(device)
        _DEPTH_MODEL = model
        print("Model Loaded : YES")
        print("Model Name : DepthAnythingV2 (vits)")
        print("Input Size : Dynamic")
        print("Output Size : Dynamic")
        print(f"Device : {device.upper()}")
        print("Inference Time : Sync")
    except Exception as e:
        print(f"Model Loaded : NO\nError: {e}")
    print("=========================\n")

_init_depth_model()


class Cfg:

    # ── ROI 
    ROI_BOTTOM_LEFT_X  = 0.10
    ROI_BOTTOM_RIGHT_X = 0.90
    ROI_TOP_LEFT_X     = 0.43
    ROI_TOP_RIGHT_X    = 0.57
    ROI_TOP_Y          = 0.65

    # ── [F] EMA cross tracking ─────────────────────────────────────
    EMA_ALPHA          = 0.15
    CROSS_BUFFER_PX    = 50
    ROI_TOP_MIN        = 0.60
    ROI_TOP_MAX        = 0.88

    # ── [G] Pentagon ROI ───────────────────────────────────────────
    USE_PENTAGON       = True
    NOTCH_TIP_Y        = 0.88
    NOTCH_HALF_W       = 0.15

    # ── [H] Auto-calibration ──────────────────────────────────────
    USE_AUTOCAL        = True
    AUTOCAL_FRAMES     = 30
    AUTOCAL_BUFFER_PX  = 60

    # ── [I] YOLO car masking ───────────────────────────────────────
    USE_YOLO           = True
    YOLO_MODEL         = "yolo11n-seg.pt"
    YOLO_CONF          = 0.35
    VEHICLE_CLASSES    = [2, 3, 5, 7]
    MASK_METHOD        = "confidence"
    CAR_BOX_PADDING    = 10

    # [P3] YOLO inference resolution
    YOLO_INFER_W       = 640
    YOLO_INFER_H       = 384

    # [P5] Async YOLO — queue depth (1 = always use freshest frame)
    YOLO_QUEUE_DEPTH   = 1

    # ── [J] Car-based ROI top ──────────────────────────────────────
    USE_CAR_ROI        = True
    CAR_ROI_BUFFER_PX  = 25
    CAR_CENTER_MIN_X   = 0.20
    CAR_CENTER_MAX_X   = 0.80

    # ── [L] BEV Geometry ───────────────────────────────────────────
    BEV_WIDTH          = 400
    BEV_HEIGHT         = 600
    ROAD_SCALE         = 2.0

    # ── [M] Metric Calibration ─────────────────────────────────────
    LANE_WIDTH_M       = 3.7

    # ── [N] Metric Occupancy Grid ──────────────────────────────────
    # Grid covers GRID_RANGE_M metres ahead and ±GRID_HALF_WIDTH_M laterally.
    # Each cell is GRID_CELL_M × GRID_CELL_M metres.
    GRID_RANGE_M       = 60.0    # metres forward
    GRID_HALF_WIDTH_M  = 7.5     # metres either side of ego
    GRID_CELL_M        = 0.25    # metres per cell
    SAFETY_MARGIN_M    = 0.6     # uniform outward expansion of footprint
    # Temporal smoothing: new_grid = α·new + (1-α)·prev
    GRID_ALPHA         = 0.55    # higher → faster response to new detections
    # Minimum track age before the footprint is trusted
    MIN_TRUST_AGE      = 3

    # ── HLS ────────────────────────────────────────────────────────
    WHITE_L_MIN        = 100
    SKY_CUT_FRAC       = 0.45
    YELLOW_H_MIN       = 10
    YELLOW_H_MAX       = 38
    YELLOW_S_MIN       = 50

    # ── Blur ───────────────────────────────────────────────────────
    BLUR_KERNEL        = (5, 5)

    # ── Hough ──────────────────────────────────────────────────────
    HOUGH_RHO          = 1
    HOUGH_THETA        = np.pi / 180
    HOUGH_THRESHOLD    = 60
    HOUGH_MIN_LENGTH   = 60
    HOUGH_MAX_GAP      = 200

    # ── Slope filter ───────────────────────────────────────────────
    LEFT_SLOPE_MIN     = -0.9
    LEFT_SLOPE_MAX     = -0.4
    RIGHT_SLOPE_MIN    =  0.4
    RIGHT_SLOPE_MAX    =  0.9
    LEFT_X_MAX_FRAC    = 0.65
    RIGHT_X_MIN_FRAC   = 0.35

    # ── Cross draw clamp ───────────────────────────────────────────
    CROSS_DRAW_BUFFER  = 30

    # ── Smoother ───────────────────────────────────────────────────
    SMOOTH_FRAMES      = 7

    # ── Drawing ────────────────────────────────────────────────────
    LANE_COLOR         = (0, 255,   0)
    FILL_COLOR         = (0, 200, 255)
    FILL_ALPHA         = 0.30
    LINE_THICKNESS     = 6
    ROI_BORDER_COLOR   = (0, 255, 100)
    CAR_BOX_COLOR      = (0,  60, 220)

    DEBUG_STREAM_SCALE = 0.5
    PRINT_BENCHMARK    = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Milestone 1 — VehicleFootprint & oriented polygon helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VehicleFootprint:
    """
    Physically correct aligned footprint for a single tracked vehicle.

    Coordinate convention (metric, BEV):
      X  – lateral  (positive = right of ego)
      Y  – longitudinal (positive = ahead of ego)

    All *_polygon fields are np.ndarray of shape (4, 2) in metres
    (footprint) or in BEV pixels (bev_*_polygon).
    """
    __slots__ = (
        'track_id', 'vehicle_class', 'confidence',
        'center_x', 'center_y',
        'length', 'width',
        'footprint_polygon',   # (4,2) metric
        'safety_polygon',      # (4,2) metric — expanded by SAFETY_MARGIN_M
        'bev_polygon',         # (4,2) pixels
        'bev_safety_polygon',  # (4,2) pixels
        'color',
        'distance_m', 'velocity_y', 'ttc', 'status',
        'lane',
    )

    def __init__(self, track_id, vehicle_class, confidence,
                 center_x, center_y,
                 length, width,
                 footprint_polygon, safety_polygon,
                 bev_polygon, bev_safety_polygon,
                 color,
                 distance_m=None, velocity_y=0.0,
                 ttc=None, status='GREEN', lane='UNKNOWN'):
        self.track_id          = track_id
        self.vehicle_class     = vehicle_class
        self.confidence        = confidence
        self.center_x          = center_x
        self.center_y          = center_y
        self.length            = length
        self.width             = width
        self.footprint_polygon = footprint_polygon
        self.safety_polygon    = safety_polygon
        self.bev_polygon       = bev_polygon
        self.bev_safety_polygon = bev_safety_polygon
        self.color             = color
        self.distance_m        = distance_m
        self.velocity_y        = velocity_y
        self.ttc               = ttc
        self.status            = status
        self.lane              = lane


def _aligned_box_polygon(cx_m, cy_m, length, width):
    """
    Return a (4,2) float32 array of the four corners of a lane-aligned
    rectangle (footprint) in the metric BEV plane.

    Args:
        cx_m, cy_m : vehicle centre in metres (BEV frame)
        length     : longitudinal size in metres
        width      : lateral size in metres
    """
    half_l = length / 2.0
    half_w = width  / 2.0
    # X is lateral (width), Y is longitudinal (length)
    corners = np.array([
        [cx_m + half_w, cy_m + half_l], # front-right
        [cx_m + half_w, cy_m - half_l], # rear-right
        [cx_m - half_w, cy_m - half_l], # rear-left
        [cx_m - half_w, cy_m + half_l], # front-left
    ], dtype=np.float32)
    return corners


def _expand_polygon(poly, margin):
    """
    Uniform outward expansion of a convex polygon by `margin` metres.
    Computes the centroid, then pushes each vertex away along the
    centroid→vertex direction.
    """
    centroid = poly.mean(axis=0)
    dirs = poly - centroid
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)   # guard zero-length
    return poly + margin * (dirs / norms)


def _metric_to_bev_pixels(poly_m, bev_w, bev_h, meters_per_pixel):
    """
    Convert a (N,2) metric polygon [X_m, Y_m] into BEV pixel coordinates.

    BEV convention (matching existing code):
      pixel_u = BEV_WIDTH/2  + X_m / meters_per_pixel   (lateral)
      pixel_v = BEV_HEIGHT   - Y_m / meters_per_pixel   (longitudinal, flipped)
    """
    ppm = 1.0 / meters_per_pixel   # pixels per metre
    px = bev_w / 2.0 + poly_m[:, 0] * ppm
    py = bev_h        - poly_m[:, 1] * ppm
    return np.stack([px, py], axis=1).astype(np.int32)


def _build_vehicle_footprint(
        track_id, cls_id, confidence,
        bev_cx, bev_cy,                # BEV pixel centre
        x_m, y_m,                      # metric centre (lateral, longitudinal)
        meters_per_pixel,
        distance_m, velocity_y,
        ttc, status, lane,
        class_names, vehicle_props,
        bev_w, bev_h):
    """
    Construct a VehicleFootprint from the available metric/pixel data.
    Returns a VehicleFootprint instance.
    """
    cls_name = class_names.get(cls_id, 'unknown')
    length_m, width_m, bgr_color = vehicle_props.get(cls_name,
                                                      vehicle_props['unknown'])

    if meters_per_pixel <= 0:
        return None

    # 1. Aligned metric footprint polygon
    fp_metric = _aligned_box_polygon(x_m, y_m, length_m, width_m)

    # 2. Safety-margin polygon (expanded outward by SAFETY_MARGIN_M)
    safe_metric = _expand_polygon(fp_metric, Cfg.SAFETY_MARGIN_M)

    # 3. Convert both to BEV pixel space
    fp_pix   = _metric_to_bev_pixels(fp_metric,   bev_w, bev_h, meters_per_pixel)
    safe_pix = _metric_to_bev_pixels(safe_metric, bev_w, bev_h, meters_per_pixel)

    return VehicleFootprint(
        track_id=track_id,
        vehicle_class=cls_name,
        confidence=confidence,
        center_x=x_m,
        center_y=y_m,
        length=length_m,
        width=width_m,
        footprint_polygon=fp_metric,
        safety_polygon=safe_metric,
        bev_polygon=fp_pix,
        bev_safety_polygon=safe_pix,
        color=bgr_color,
        distance_m=distance_m,
        velocity_y=velocity_y,
        ttc=ttc,
        status=status,
        lane=lane,
    )


# Occupancy grid state (persisted across frames for temporal smoothing)
_occ_grid_rows = max(1, int(Cfg.GRID_RANGE_M    / Cfg.GRID_CELL_M))
_occ_grid_cols = max(1, int(Cfg.GRID_HALF_WIDTH_M * 2 / Cfg.GRID_CELL_M))
_occ_grid = np.zeros((_occ_grid_rows, _occ_grid_cols), dtype=np.float32)


def _rasterize_footprints_to_grid(footprints, meters_per_pixel):
    """
    Rasterize all vehicle safety polygons into the metric occupancy grid.

    Grid indexing:
      row 0            = farthest forward (GRID_RANGE_M ahead)
      row GRID_ROWS-1  = ego front bumper (0 m)
      col 0            = leftmost  (-GRID_HALF_WIDTH_M)
      col GRID_COLS-1  = rightmost (+GRID_HALF_WIDTH_M)

    Returns: new uint8 grid (255=occupied, 0=free) and the temporally
    smoothed float32 grid.
    """
    global _occ_grid

    rows = _occ_grid_rows
    cols = _occ_grid_cols
    cell = Cfg.GRID_CELL_M
    half_w = Cfg.GRID_HALF_WIDTH_M
    rng    = Cfg.GRID_RANGE_M

    new_grid = np.zeros((rows, cols), dtype=np.float32)

    for fp in footprints:
        if fp is None:
            continue
        # Convert metric safety polygon corners → grid cell indices
        # X_m → col:  col = (X_m + half_w) / cell
        # Y_m → row:  row = (rng - Y_m)    / cell   (row 0 = farthest)
        poly_m = fp.safety_polygon  # (4,2): X_m, Y_m
        col_f  = (poly_m[:, 0] + half_w) / cell
        row_f  = (rng - poly_m[:, 1])    / cell
        grid_poly = np.stack([col_f, row_f], axis=1).astype(np.int32)
        # fillPoly expects (N,1,2) in (x=col, y=row) order
        grid_poly_cv = grid_poly[:, np.newaxis, :]
        cv2.fillPoly(new_grid, [grid_poly_cv], 1.0)

    # Temporal smoothing
    _occ_grid = Cfg.GRID_ALPHA * new_grid + (1.0 - Cfg.GRID_ALPHA) * _occ_grid
    return (_occ_grid >= 0.35).astype(np.uint8) * 255


def _draw_footprint_on_bev(bev_img, fp: VehicleFootprint):
    """
    Render a VehicleFootprint onto the BEV image.
    Draws: filled footprint polygon, safety-margin outline,
    ID + distance text.
    """
    color     = fp.color
    safe_col  = tuple(max(0, c - 60) for c in color)   # slightly darker

    # Filled vehicle footprint
    cv2.fillPoly(bev_img, [fp.bev_polygon.reshape(-1, 1, 2)], color)
    # White outline of actual footprint
    cv2.polylines(bev_img, [fp.bev_polygon.reshape(-1, 1, 2)],
                  True, (255, 255, 255), 1, cv2.LINE_AA)
    # Dashed-effect safety margin (draw outline in translucent darker tone)
    cv2.polylines(bev_img, [fp.bev_safety_polygon.reshape(-1, 1, 2)],
                  True, safe_col, 1, cv2.LINE_AA)

    # TTC-based status colour for label
    lbl_col = (0, 255, 0)
    if fp.status == 'RED':
        lbl_col = (0, 0, 255)
    elif fp.status == 'YELLOW':
        lbl_col = (0, 220, 255)

    label_x = fp.bev_polygon[:, 0].min()
    label_y = fp.bev_polygon[:, 1].min() - 16
    cv2.putText(bev_img,
                f"ID{fp.track_id}  {fp.distance_m:.1f}m" if fp.distance_m is not None
                else f"ID{fp.track_id}",
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, lbl_col, 1, cv2.LINE_AA)
    cls_short = fp.vehicle_class[:3].upper()
    cv2.putText(bev_img, cls_short,
                (int(label_x), int(label_y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)


def get_vehicle_contact_point(mask):
    """
    Finds the lowest pixel (max y) of the given binary mask.
    Returns (contact_x, contact_y) or (None, None) if mask is empty.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None, None
    max_y_idx = np.argmax(y_indices)
    return int(x_indices[max_y_idx]), int(y_indices[max_y_idx])

class _ContactPointSmoother:
    def __init__(self, maxlen=5):
        self.history = {}
        self.maxlen = maxlen

    def update(self, track_id, pt):
        if pt[0] is None or pt[1] is None:
            return None, None, 0, 0
        
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.maxlen)
            
        pts = self.history[track_id]
        prev_x = sum(p[0] for p in pts) / len(pts) if pts else pt[0]
        prev_y = sum(p[1] for p in pts) / len(pts) if pts else pt[1]

        self.history[track_id].append(pt)
        
        pts = self.history[track_id]
        avg_x = sum(p[0] for p in pts) / len(pts)
        avg_y = sum(p[1] for p in pts) / len(pts)
        
        dx = int(avg_x - prev_x)
        dy = int(avg_y - prev_y)
        
        return int(avg_x), int(avg_y), dx, dy

    def cleanup(self, active_ids):
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]

class _AsyncCarDetector:
    """
    Keeps a background thread that runs YOLO inference.
    The pipeline thread pushes frames into _in_q and reads
    the latest boxes from _last_boxes — no blocking wait.

    If YOLO is disabled or unavailable, detect() always returns [], [].
    """

    def __init__(self):
        self._model       = None
        self._last_boxes  = []
        self._last_masks  = []
        self._boxes_lock  = threading.Lock()
        self._in_q        = queue.Queue(maxsize=Cfg.YOLO_QUEUE_DEPTH)
        self._frame_idx   = 0
        self._thread      = None
        self._started     = False
        self._smoother    = _ContactPointSmoother(maxlen=5)

    # ── start the worker once ─────────────────────────────────────
    def _ensure_started(self):
        if self._started:
            return
        self._started = True
        if not Cfg.USE_YOLO or not _YOLO_AVAILABLE:
            return
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="yolo-worker"
        )
        self._thread.start()

    def _worker(self):
        """Background thread: load model once, then consume frames."""
        print("[YOLO] Loading model…")
        self._model = _YOLO(Cfg.YOLO_MODEL)
        print("[YOLO] Ready — async thread running")

        while True:
            try:
                frame = self._in_q.get(timeout=5.0)
            except queue.Empty:
                continue

            if frame is None:       # sentinel — shut down
                break

            h_orig, w_orig = frame.shape[:2]
            infer = cv2.resize(
                frame, (Cfg.YOLO_INFER_W, Cfg.YOLO_INFER_H),
                interpolation=cv2.INTER_LINEAR,
            )

            results = self._model.track(
                infer,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=Cfg.YOLO_CONF,
                classes=Cfg.VEHICLE_CLASSES,
                imgsz=(Cfg.YOLO_INFER_H, Cfg.YOLO_INFER_W),
            )

            boxes = results[0].boxes
            masks = results[0].masks
            
            if boxes is None or len(boxes) == 0:
                with self._boxes_lock:
                    self._last_boxes = []
                    self._last_masks = []
                self._smoother.cleanup(set())
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls_ids = boxes.cls.cpu().numpy().reshape(-1, 1)
            
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().reshape(-1, 1)
            else:
                track_ids = np.full((len(boxes), 1), -1.0)

            sx   = w_orig / Cfg.YOLO_INFER_W
            sy   = h_orig / Cfg.YOLO_INFER_H
            xyxy_scaled = xyxy * np.array([sx, sy, sx, sy])

            contacts_x = np.full((len(boxes), 1), -1.0)
            contacts_y = np.full((len(boxes), 1), -1.0)
            contacts_dx = np.full((len(boxes), 1), 0.0)
            contacts_dy = np.full((len(boxes), 1), 0.0)
            
            mask_list = []
            active_ids = set()
            
            if masks is not None:
                mask_data = masks.data.cpu().numpy()
                
                for i in range(len(boxes)):
                    tid = float(track_ids[i, 0])
                    if tid != -1.0:
                        active_ids.add(tid)
                        
                    mask_resized = cv2.resize(mask_data[i], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    mask_list.append(mask_resized)
                    
                    cx, cy = get_vehicle_contact_point(mask_resized)
                    if cx is not None and cy is not None and tid != -1.0:
                        scx, scy, dx, dy = self._smoother.update(tid, (cx, cy))
                        if scx is not None and scy is not None:
                            contacts_x[i, 0] = float(scx)
                            contacts_y[i, 0] = float(scy)
                            contacts_dx[i, 0] = float(dx)
                            contacts_dy[i, 0] = float(dy)
            else:
                for i in range(len(boxes)):
                    mask_list.append(None)
                    
            self._smoother.cleanup(active_ids)

            with self._boxes_lock:
                self._last_boxes = np.hstack([xyxy_scaled, conf, cls_ids, track_ids, contacts_x, contacts_y, contacts_dx, contacts_dy]).tolist()
                self._last_masks = mask_list

    def detect(self, frame):
        """
        Non-blocking: submits frame to worker every YOLO_EVERY_N frames,
        always returns the latest cached boxes immediately.
        """
        self._ensure_started()
        self._frame_idx += 1

        if not Cfg.USE_YOLO or not _YOLO_AVAILABLE:
            return [], []

        if self._frame_idx % 1 == 0:
            try:
                self._in_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        with self._boxes_lock:
            return list(self._last_boxes), list(self._last_masks)

    def stop(self):
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass



def mask_cars(lane_mask, car_boxes):
    if not car_boxes:
        return lane_mask

    result = lane_mask.copy()
    pad    = Cfg.CAR_BOX_PADDING
    h, w   = lane_mask.shape[:2]

    for box in car_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]) if len(box) > 4 else 1.0
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

        if Cfg.MASK_METHOD == "direct":
            result[y1:y2, x1:x2] = 0
        elif Cfg.MASK_METHOD == "erode":
            bw = x2 - x1; bh = y2 - y1
            ex = int(bw * 0.08); ey = int(bh * 0.08)
            result[y1+ey:y2-ey, x1+ex:x2-ex] = 0
        elif Cfg.MASK_METHOD == "confidence":
            if conf >= 0.70:
                result[y1:y2, x1:x2] = 0
            elif conf >= 0.40:
                cx = (x1+x2)//2; cy = (y1+y2)//2
                hw = (x2-x1)//4; hh = (y2-y1)//4
                result[cy-hh:cy+hh, cx-hw:cx+hw] = 0

    return result

def car_based_roi_top(car_boxes, frame_h, frame_w):
    if not Cfg.USE_CAR_ROI or not car_boxes:
        return None
    min_x = frame_w * Cfg.CAR_CENTER_MIN_X
    max_x = frame_w * Cfg.CAR_CENTER_MAX_X
    bottoms = [
        int(box[3]) for box in car_boxes
        if min_x < (int(box[0]) + int(box[2])) / 2 < max_x
    ]
    if not bottoms:
        return None
    roi_y = (max(bottoms) + Cfg.CAR_ROI_BUFFER_PX) / frame_h
    return float(np.clip(roi_y, Cfg.ROI_TOP_MIN, Cfg.ROI_TOP_MAX))



def hls_lane_mask(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H   = hls[:, :, 0].astype(np.int32)
    L   = hls[:, :, 1].astype(np.int32)
    S   = hls[:, :, 2].astype(np.int32)

    white_mask  = (L >= Cfg.WHITE_L_MIN)
    yellow_mask = (
        (H >= Cfg.YELLOW_H_MIN) & (H <= Cfg.YELLOW_H_MAX) &
        (S >= Cfg.YELLOW_S_MIN)
    )

    sky_cut = np.ones_like(L, dtype=np.uint8)
    sky_cut[:int(L.shape[0] * Cfg.SKY_CUT_FRAC), :] = 0

    lane_mask = np.clip(
        white_mask.astype(np.uint8) + yellow_mask.astype(np.uint8), 0, 1
    ).astype(np.uint8)
    lane_mask = (lane_mask * sky_cut * 255).astype(np.uint8)

    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, lane_mask)

    return masked_gray, lane_mask, gray


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Otsu Canny
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def otsu_canny(gray_image):
    blurred = cv2.GaussianBlur(gray_image, Cfg.BLUR_KERNEL, 0)
    otsu_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_low  = max(10, int(otsu_val * 0.5))
    t_high = max(20, int(otsu_val))
    return cv2.Canny(blurred, t_low, t_high)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _EMA:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if new_val is None:
            return self.value
        if self.value is None:
            self.value = float(new_val)
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Auto-calibrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _AutoCalibrator:
    def __init__(self):
        self._samples = []
        self._locked  = False
        self._roi_top = Cfg.ROI_TOP_Y

    @property
    def roi_top(self): return self._roi_top

    @property
    def is_locked(self): return self._locked

    def update(self, cross_y, frame_h):
        if self._locked:
            return self._roi_top
        if cross_y is not None:
            self._samples.append(cross_y)
        if len(self._samples) >= Cfg.AUTOCAL_FRAMES:
            med           = float(np.median(self._samples))
            self._roi_top = float(np.clip(
                (med + Cfg.AUTOCAL_BUFFER_PX) / frame_h,
                Cfg.ROI_TOP_MIN, Cfg.ROI_TOP_MAX))
            self._locked  = True
            print(f"[AutoCal] Locked — roi_top={self._roi_top:.3f} "
                  f"(median cross_y={med:.0f}px)")
        return self._roi_top


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pentagon / trapezoid ROI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_roi_polygon(h, w, top_y_frac):
    top_y   = int(h * top_y_frac)
    cx      = w // 2
    tip_y   = int(h * Cfg.NOTCH_TIP_Y)
    half_nw = int(w * Cfg.NOTCH_HALF_W)

    if Cfg.USE_PENTAGON:
        return np.array([[
            (int(w * Cfg.ROI_BOTTOM_LEFT_X),  h    ),
            (int(w * Cfg.ROI_TOP_LEFT_X),     top_y),
            (int(w * Cfg.ROI_TOP_RIGHT_X),    top_y),
            (int(w * Cfg.ROI_BOTTOM_RIGHT_X), h    ),
            (cx + half_nw,                    h    ),
            (cx,                              tip_y),
            (cx - half_nw,                    h    ),
        ]], dtype=np.int32)
    return np.array([[
        (int(w * Cfg.ROI_BOTTOM_LEFT_X),  h    ),
        (int(w * Cfg.ROI_TOP_LEFT_X),     top_y),
        (int(w * Cfg.ROI_TOP_RIGHT_X),    top_y),
        (int(w * Cfg.ROI_BOTTOM_RIGHT_X), h    ),
    ]], dtype=np.int32)


def apply_roi(edges, h, w, top_y_frac):
    polygon = build_roi_polygon(h, w, top_y_frac)
    mask    = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked  = cv2.bitwise_and(edges, mask)
    roi_vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    return masked, roi_vis, polygon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RANSAC fit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fit(lines_group):
    if len(lines_group) < 2 or not _RANSAC_AVAILABLE:
        return tuple(np.mean(lines_group, axis=0))
    slopes     = np.array([s for s, _ in lines_group]).reshape(-1, 1)
    intercepts = np.array([b for _, b in lines_group])
    try:
        reg     = RANSACRegressor(min_samples=2, residual_threshold=10)
        reg.fit(slopes, intercepts)
        inliers = [lines_group[i]
                   for i, ok in enumerate(reg.inlier_mask_) if ok]
        return tuple(np.mean(inliers, axis=0)) if inliers else \
               tuple(np.mean(lines_group, axis=0))
    except Exception:
        return tuple(np.mean(lines_group, axis=0))


#
def filter_and_fit(lines, h, w, roi_top_frac):
    ls, li, rs, ri = [], [], [], []
    if lines is None:
        return None, None, None

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if x2 == x1: continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        if Cfg.LEFT_SLOPE_MIN < m < Cfg.LEFT_SLOPE_MAX:
            if x1 < w * Cfg.LEFT_X_MAX_FRAC:
                ls.append(m); li.append(b)
        elif Cfg.RIGHT_SLOPE_MIN < m < Cfg.RIGHT_SLOPE_MAX:
            if x1 > w * Cfg.RIGHT_X_MIN_FRAC:
                rs.append(m); ri.append(b)

    lm, lb = _fit(list(zip(ls, li))) if ls else (None, None)
    rm, rb = _fit(list(zip(rs, ri))) if rs else (None, None)

    cross_y = None
    if lm and rm and abs(lm - rm) > 1e-6:
        xc      = (rb - lb) / (lm - rm)
        cross_y = int(lm * xc + lb)

    def coords(m, b):
        if m is None: return None
        y1c = h
        y2c = int(h * roi_top_frac)
        if cross_y is not None:
            y2c = max(y2c, cross_y + Cfg.CROSS_DRAW_BUFFER)
        return (int((y1c - b) / m), y1c, int((y2c - b) / m), y2c)

    return coords(lm, lb), coords(rm, rb), cross_y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Temporal smoother
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Smoother:
    def __init__(self, n=7):
        self._L = deque(maxlen=n)
        self._R = deque(maxlen=n)

    def update(self, left, right):
        if left  is not None: self._L.append(left)
        if right is not None: self._R.append(right)
        def avg(buf):
            if not buf: return None
            return tuple(int(v) for v in np.mean(buf, axis=0))
        return avg(self._L), avg(self._R)



def build_hls_vis(lane_mask, car_boxes):
    vis = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    for box in car_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]) if len(box) > 4 else 1.0
        cv2.rectangle(vis, (x1, y1), (x2, y2), Cfg.CAR_BOX_COLOR, 2)
        text = f"{conf:.2f}"
        if len(box) > 5 and box[5] != -1:
            text = f"ID:{int(box[5])} {text}"
        cv2.putText(vis, text, (x1+3, y1+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, Cfg.CAR_BOX_COLOR, 1)
    return vis



_FILL_COLOR_NP = np.array(Cfg.FILL_COLOR, dtype=np.uint8)

def draw_overlay(frame, left, right, roi_polygon, calibrated, car_boxes, car_masks=None, src_pts=None, lane_pts=None):
    out = frame.copy()

    if lane_pts is not None:
        pts = np.array([
            lane_pts[0], # TL
            lane_pts[1], # TR
            lane_pts[3], # BR
            lane_pts[2]  # BL
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        
        cv2.putText(out, "TL", (int(lane_pts[0][0]), int(lane_pts[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(out, "TR", (int(lane_pts[1][0]), int(lane_pts[1][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(out, "BL", (int(lane_pts[2][0]), int(lane_pts[2][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(out, "BR", (int(lane_pts[3][0]), int(lane_pts[3][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    if src_pts is not None:
        for pt in src_pts:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
        
        pts = np.array([
            src_pts[0], # TL
            src_pts[1], # TR
            src_pts[3], # BR
            src_pts[2]  # BL
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (255, 0, 0), 2)
        
        cv2.putText(out, "TL", (int(src_pts[0][0]), int(src_pts[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(out, "TR", (int(src_pts[1][0]), int(src_pts[1][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(out, "BL", (int(src_pts[2][0]), int(src_pts[2][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(out, "BR", (int(src_pts[3][0]), int(src_pts[3][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if car_masks is not None:
        mask_overlay = np.zeros_like(out)
        for i, mask in enumerate(car_masks):
            if mask is not None:
                color = Cfg.CAR_BOX_COLOR
                mask_overlay[mask > 0] = color
        
        alpha = 0.4
        mask_bool = np.any(mask_overlay > 0, axis=-1)
        out[mask_bool] = (out[mask_bool] * (1 - alpha) + mask_overlay[mask_bool] * alpha).astype(np.uint8)

    if left is not None and right is not None:
        pts = np.array([
            [left[0],  left[1]], [left[2],  left[3]],
            [right[2], right[3]], [right[0], right[1]],
        ], dtype=np.int32)

        # [P7] Build mask once, blend with np.where — one allocation less
        poly_mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        alpha = Cfg.FILL_ALPHA
        mask3 = poly_mask[:, :, np.newaxis].astype(np.float32) / 255.0
        out   = (out.astype(np.float32) * (1 - mask3 * alpha) +
                 _FILL_COLOR_NP.astype(np.float32) * (mask3 * alpha)
                 ).astype(np.uint8)

    for ln in (left, right):
        if ln is not None:
            cv2.line(out, (ln[0], ln[1]), (ln[2], ln[3]),
                     Cfg.LANE_COLOR, Cfg.LINE_THICKNESS)

    for box in car_boxes:
        if len(box) > 12 and box[7] != -1.0 and box[8] != -1.0:
            cx, cy = int(box[7]), int(box[8])
            lane_side = box[11]
            validity = box[12]
            
            if validity == "VALID":
                color = (0, 0, 255) # Red
            else:
                color = (128, 128, 128) # Gray
                
            cv2.circle(out, (cx, cy), 6, color, -1)
            
            tid = int(box[6]) if box[6] != -1 else -1
            if tid != -1:
                cv2.putText(out, f"[{tid}]", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if isinstance(lane_side, str):
                    cv2.putText(out, lane_side, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    roi_color = (0, 200, 0) if calibrated else (0, 180, 255)
    cv2.polylines(out, roi_polygon, True, roi_color, 1)
    status = f"CAL | cars={len(car_boxes)}" if calibrated else f"WARM | cars={len(car_boxes)}"
    cv2.putText(out, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
    return out


def assign_lane(cx, cy, left_line, right_line, w):
    if cx is None or cy is None or cx == -1 or cy == -1:
        return "UNKNOWN"
        
    def get_x_at_y(line, y_target):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if y1 == y2:
            return None
        return x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)

    left_x = get_x_at_y(left_line, cy)
    right_x = get_x_at_y(right_line, cy)

    if left_x is None and right_x is None:
        left_x = w * 0.33
        right_x = w * 0.66
    elif left_x is None:
        left_x = right_x - (w * 0.33)
    elif right_x is None:
        right_x = left_x + (w * 0.33)

    if cx < left_x:
        return "LEFT"
    elif cx > right_x:
        return "RIGHT"
    else:
        return "CENTER"


class _HomographyTracker:
    def __init__(self, calib_frames=30):
        self.H_fixed = None
        self.fixed_src_pts = None
        self.current_lane_pts = None
        self.current_src_pts = None
        self.widths = (0, 0, 0, 0)
        
        self.calib_frames = calib_frames
        self.calib_samples = []
        self.calib_count = 0

    def update(self, left_line, right_line, w):
        if left_line is not None and right_line is not None:
            lb_x, lb_y, lt_x, lt_y = left_line
            rb_x, rb_y, rt_x, rt_y = right_line
            
            top_y = min(lt_y, rt_y)
            lt_y = top_y
            rt_y = top_y
            
            bot_y = max(lb_y, rb_y)
            lb_y = bot_y
            rb_y = bot_y
            
            lane_w_top = rt_x - lt_x
            lane_w_bot = rb_x - lb_x
            
            road_lt_x = max(0, lt_x - Cfg.ROAD_SCALE * lane_w_top)
            road_rt_x = min(w - 1, rt_x + Cfg.ROAD_SCALE * lane_w_top)
            
            road_top_width = road_rt_x - road_lt_x
            if road_top_width > 1200:
                diff = road_top_width - 1200
                road_lt_x += diff / 2
                road_rt_x -= diff / 2
            
            road_lb_x = max(0, lb_x - Cfg.ROAD_SCALE * lane_w_bot)
            road_rb_x = min(w - 1, rb_x + Cfg.ROAD_SCALE * lane_w_bot)
            
            self.widths = (lane_w_top, lane_w_bot, road_rt_x - road_lt_x, road_rb_x - road_lb_x)

            src = np.array([
                [road_lt_x, top_y],
                [road_rt_x, top_y],
                [road_lb_x, bot_y],
                [road_rb_x, bot_y]
            ], dtype=np.float32)
            
            lane_pts = np.array([
                [lt_x, lt_y],
                [rt_x, rt_y],
                [lb_x, lb_y],
                [rb_x, rb_y]
            ], dtype=np.float32)

            self.current_src_pts = src
            self.current_lane_pts = lane_pts

            if self.calib_count < self.calib_frames:
                self.calib_samples.append(src)
                self.calib_count += 1
                
                if self.calib_count == self.calib_frames:
                    median_src = np.median(np.array(self.calib_samples), axis=0).astype(np.float32)
                    self.fixed_src_pts = median_src
                    
                    dst = np.array([
                        [0, 0],
                        [Cfg.BEV_WIDTH, 0],
                        [0, Cfg.BEV_HEIGHT],
                        [Cfg.BEV_WIDTH, Cfg.BEV_HEIGHT]
                    ], dtype=np.float32)
                    
                    self.H_fixed = cv2.getPerspectiveTransform(median_src, dst)
                    print(f"\n[HomographyTracker] Calibration complete. H_fixed locked with {self.calib_frames} samples.\n")

        if self.H_fixed is None:
            return None, self.current_src_pts, self.current_lane_pts
        else:
            return self.H_fixed, self.fixed_src_pts, self.current_lane_pts

_smoother   = _Smoother(Cfg.SMOOTH_FRAMES)
_ema_cross  = _EMA(alpha=Cfg.EMA_ALPHA)
_calibrator = _AutoCalibrator()
_detector   = _AsyncCarDetector()          
_homography = _HomographyTracker()
_track_states = {}



from enum import Enum

class DrivingState(Enum):
    KEEP_LANE = "KEEP_LANE"
    FOLLOW_LEAD = "FOLLOW_LEAD"
    CHANGE_LEFT = "CHANGE_LEFT"
    CHANGE_RIGHT = "CHANGE_RIGHT"
    PREPARE_LEFT = "PREPARE_LEFT"
    PREPARE_RIGHT = "PREPARE_RIGHT"
    SLOW_DOWN = "SLOW_DOWN"
    STOP = "STOP"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    UNKNOWN = "UNKNOWN"

class BehaviorPlanner:
    def plan(self, lane_occupancy, free_space, gap_detection, lane_scores, lead_vehicle):
        if lead_vehicle is not None and lead_vehicle[13] < 1.5:
            return DrivingState.EMERGENCY_BRAKE, "Obstacle closer than 1.5 m.", 99
            
        if lead_vehicle is not None and lead_vehicle[13] < 3.0:
            return DrivingState.STOP, "Obstacle closer than 3.0 m.", 95
            
        lead_dist = lead_vehicle[13] if lead_vehicle is not None else 30.0
        
        if lead_dist < 10.0:
            left_safe = len([c for c in lane_occupancy["LEFT"] if c[13] < 10.0]) == 0
            right_safe = len([c for c in lane_occupancy["RIGHT"] if c[13] < 10.0]) == 0
            
            left_gap = gap_detection.get("LEFT", 0.0)
            right_gap = gap_detection.get("RIGHT", 0.0)
            
            if left_safe and left_gap > right_gap:
                return DrivingState.CHANGE_LEFT, "Lead vehicle too close. Left lane has larger free gap.", 92
            elif right_safe and right_gap >= left_gap:
                return DrivingState.CHANGE_RIGHT, "Lead vehicle too close. Right lane has larger free gap.", 92
            else:
                return DrivingState.SLOW_DOWN, "No safe lane change possible.", 89
                
        if lead_dist < 30.0:
            return DrivingState.FOLLOW_LEAD, "Lead vehicle in following range.", 87
            
        return DrivingState.KEEP_LANE, "Center lane free. Road ahead clear.", 96

_planner = BehaviorPlanner()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_print_frame_counter = 0

_bev_history = {}

def detect_lanes(frame):
    global _print_frame_counter
    _print_frame_counter += 1
    
    current_time = time.time()
    
    if Cfg.PRINT_BENCHMARK:
        _t0 = time.perf_counter()

    h, w = frame.shape[:2]

    car_boxes, car_masks = _detector.detect(frame)
    
    debug_frames = {}
    print("YOLO Detection\n↓")

    depth_map = None
    if _DEPTH_MODEL is not None:
        depth_map = _DEPTH_MODEL.infer_image(frame)
        print("Depth Map\n↓")
        d_min, d_max = depth_map.min(), depth_map.max()
        d_mean, d_std = depth_map.mean(), depth_map.std()
        print("Depth Statistics")
        print(f"Min Depth : {d_min:.2f}")
        print(f"Max Depth : {d_max:.2f}")
        print(f"Mean Depth : {d_mean:.2f}")
        print(f"Std : {d_std:.2f}\n")
        
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        debug_frames["Depth Debug"] = depth_colored
        debug_frames["RGB + Depth Overlay"] = frame.copy()
        debug_frames["3D Box Debug"] = frame.copy()
    else:
        print("PIPELINE BROKEN AT:\nDepth Map\n")

    masked_gray, raw_mask, gray = hls_lane_mask(frame)

    clean_mask  = mask_cars(raw_mask, car_boxes)
    masked_gray = cv2.bitwise_and(gray, clean_mask)

    edges = otsu_canny(masked_gray)

    car_top = car_based_roi_top(car_boxes, h, w)
    if Cfg.USE_AUTOCAL:
        cal_top = _calibrator.update(_ema_cross.value, h)
    else:
        ema_val = _ema_cross.value
        cal_top = (
            float(np.clip((ema_val + Cfg.CROSS_BUFFER_PX) / h,
                          Cfg.ROI_TOP_MIN, Cfg.ROI_TOP_MAX))
            if ema_val is not None else Cfg.ROI_TOP_Y
        )
    top_y_frac = max(cal_top, car_top) if car_top is not None else cal_top

    roi_masked, roi_vis, roi_poly = apply_roi(edges, h, w, top_y_frac)

    # Hough
    lines = cv2.HoughLinesP(
        roi_masked,
        rho=Cfg.HOUGH_RHO, theta=Cfg.HOUGH_THETA,
        threshold=Cfg.HOUGH_THRESHOLD,
        minLineLength=Cfg.HOUGH_MIN_LENGTH,
        maxLineGap=Cfg.HOUGH_MAX_GAP,
    )

    left_raw, right_raw, cross_y = filter_and_fit(lines, h, w, top_y_frac)

    _ema_cross.update(cross_y)
    if Cfg.USE_AUTOCAL:
        _calibrator.update(cross_y, h)

    left, right = _smoother.update(left_raw, right_raw)

    H, src_pts, lane_pts = _homography.update(left, right, w)
    
    lane_w_top, lane_w_bot, road_w_top, road_w_bot = _homography.widths
    print(f"Lane Top Width = {int(lane_w_top)} px")
    print(f"Lane Bottom Width = {int(lane_w_bot)} px")
    print(f"Road Top Width = {int(road_w_top)} px")
    print(f"Road Bottom Width = {int(road_w_bot)} px\n")
    
    if _homography.current_src_pts is not None:
        c_pts = _homography.current_src_pts
        print("Current Source Corridor:")
        print(f"ROAD_TL=({int(c_pts[0][0])},{int(c_pts[0][1])})")
        print(f"ROAD_TR=({int(c_pts[1][0])},{int(c_pts[1][1])})")
        print(f"ROAD_BL=({int(c_pts[2][0])},{int(c_pts[2][1])})")
        print(f"ROAD_BR=({int(c_pts[3][0])},{int(c_pts[3][1])})\n")

    if _homography.fixed_src_pts is not None:
        f_pts = _homography.fixed_src_pts
        print("Fixed Projected Corridor:")
        print(f"ROAD_TL=({int(f_pts[0][0])},{int(f_pts[0][1])})")
        print(f"ROAD_TR=({int(f_pts[1][0])},{int(f_pts[1][1])})")
        print(f"ROAD_BL=({int(f_pts[2][0])},{int(f_pts[2][1])})")
        print(f"ROAD_BR=({int(f_pts[3][0])},{int(f_pts[3][1])})\n")
    
    homography_engine = None
    try:
        from phase_02_geometry.homography import Homography
        homography_engine = Homography()
    except Exception as e:
        print("Failed to load Phase 2 Homography:", e)

    if homography_engine is not None and homography_engine.H_inv is not None:
        det_H = np.linalg.det(homography_engine.H_inv)
        print(f"det(H_inv)={det_H:.6f}")
            
        pts_to_proj = np.array([[[pt[0], pt[1]]] for pt in lane_pts], dtype=np.float32)
        proj_lane_pts = homography_engine.project_points(pts_to_proj)
        
        if proj_lane_pts is not None and len(proj_lane_pts) > 0:
            p_tl, p_tr, p_bl, p_br = proj_lane_pts[:, 0, :]
            lane_w_px_top = p_tr[0] - p_tl[0]
            lane_w_px_bot = p_br[0] - p_bl[0]
            lane_w_px = (lane_w_px_top + lane_w_px_bot) / 2.0
            
            meters_per_pixel = homography_engine.MPP
            
            print(f"Lane Width BEV = {lane_w_px:.1f} px")
            print(f"Meters Per Pixel = {meters_per_pixel:.4f} m/px\n")
        else:
            meters_per_pixel = homography_engine.MPP
    else:
        meters_per_pixel = 0.0

    # Phase 7 - Drivable Corridor Fill
    bev_vis = np.zeros((Cfg.BEV_HEIGHT, Cfg.BEV_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(bev_vis, (0, 0), (Cfg.BEV_WIDTH, Cfg.BEV_HEIGHT), (0, 40, 0), -1)
    
    if src_pts is not None and homography_engine is not None:
        pts_to_project = np.array([[[pt[0], pt[1]]] for pt in lane_pts], dtype=np.float32)
        proj_lane_pts = homography_engine.project_points(pts_to_project)
        if proj_lane_pts is not None and len(proj_lane_pts) > 0:
            p_tl, p_tr, p_bl, p_br = proj_lane_pts[:, 0, :]
            
            cv2.line(bev_vis, (int(p_tl[0]), int(p_tl[1])), (int(p_bl[0]), int(p_bl[1])), (0, 255, 0), 2)
            cv2.line(bev_vis, (int(p_tr[0]), int(p_tr[1])), (int(p_br[0]), int(p_br[1])), (0, 255, 0), 2)
            
            cx_top = int((p_tl[0] + p_tr[0]) / 2)
            cx_bot = int((p_bl[0] + p_br[0]) / 2)
            cv2.line(bev_vis, (cx_top, int(p_tl[1])), (cx_bot, int(p_bl[1])), (255, 255, 255), 1)

    lane_counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    decorated_boxes = []
    
    inside_count = 0
    outside_count = 0
    
    CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    VEHICLE_PROPS = {
        "car": (4.5, 1.8, (255, 0, 0)),
        "truck": (10.0, 2.5, (0, 165, 255)),
        "bus": (12.0, 2.6, (128, 0, 128)),
        "motorcycle": (2.2, 0.8, (0, 255, 255)),
        "unknown": (4.5, 1.8, (128, 128, 128))
    }
    
    if meters_per_pixel > 0:
        ego_len_px = int(4.5 / meters_per_pixel)
        ego_wid_px = int(1.8 / meters_per_pixel)
        ex = Cfg.BEV_WIDTH // 2
        ey = Cfg.BEV_HEIGHT - ego_len_px // 2
        cv2.rectangle(bev_vis, (ex - ego_wid_px//2, ey - ego_len_px//2),
                      (ex + ego_wid_px//2, ey + ego_len_px//2), (128, 128, 128), -1)
        cv2.rectangle(bev_vis, (ex - ego_wid_px//2, ey - ego_len_px//2),
                      (ex + ego_wid_px//2, ey + ego_len_px//2), (255, 255, 255), 2)
        cv2.putText(bev_vis, "EGO", (ex - 15, ey + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for box in car_boxes:
        cls_id = int(box[5]) if len(box) > 5 else 2
        tid = int(box[6]) if len(box) > 6 and box[6] != -1 else None
        cx = int(box[7]) if len(box) > 7 and box[7] != -1 else None
        cy = int(box[8]) if len(box) > 8 and box[8] != -1 else None
        dx = int(box[9]) if len(box) > 9 else 0
        dy = int(box[10]) if len(box) > 10 else 0
        
        if tid is not None and cx is not None and cy is not None:
            validity = "VALID"
            lane_side = assign_lane(cx, cy, left, right, w)
            
            print(f"==================================\nVehicle ID : {tid}")
            print(f"YOLO Class : {cls_id}\nBBox")
            print(f"x1 : {int(box[0])}\ny1 : {int(box[1])}\nx2 : {int(box[2])}\ny2 : {int(box[3])}")
            
            vehicle_depth = None
            if depth_map is not None:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                roi_depth = depth_map[y1:y2, x1:x2]
                if roi_depth.size > 0:
                    med_d = np.median(roi_depth)
                    mean_d = np.mean(roi_depth)
                    min_d = np.min(roi_depth)
                    max_d = np.max(roi_depth)
                    std_d = np.std(roi_depth)
                    vehicle_depth = med_d
                    print("Depth Pixels")
                    print(f"Total Pixels : {roi_depth.size}")
                    print(f"Median Depth : {med_d:.2f}")
                    print(f"Mean Depth : {mean_d:.2f}")
                    print(f"Minimum : {min_d:.2f}")
                    print(f"Maximum : {max_d:.2f}")
                    print(f"Depth Std : {std_d:.2f}\n")
                    
                    cv2.putText(debug_frames["RGB + Depth Overlay"], f"ID={tid}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(debug_frames["RGB + Depth Overlay"], f"Depth={med_d:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(debug_frames["RGB + Depth Overlay"], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    print("Depth Value\n↓")
                    
                    fx, fy = w, w
                    u0, v0 = w/2, h/2
                    Z = med_d
                    X_cam = (cx - u0) * Z / fx
                    Y_cam = (cy - v0) * Z / fy
                    print("Camera Coordinates")
                    print(f"X : {X_cam:.2f}")
                    print(f"Y : {Y_cam:.2f}")
                    print(f"Z : {Z:.2f}\n")
                    
                    print("Camera Coordinates\n↓")
                    
                    length_m, width_m, _ = VEHICLE_PROPS.get(CLASS_NAMES.get(cls_id, 'unknown'), VEHICLE_PROPS['unknown'])
                    height_m = 1.6
                    half_l, half_w = length_m/2, width_m/2
                    corners_3d = np.array([
                        [-half_w, -height_m, half_l], [half_w, -height_m, half_l],
                        [half_w, 0, half_l], [-half_w, 0, half_l],
                        [-half_w, -height_m, -half_l], [half_w, -height_m, -half_l],
                        [half_w, 0, -half_l], [-half_w, 0, -half_l]
                    ])
                    corners_cam = corners_3d + np.array([X_cam, Y_cam, Z])
                    proj_x = (corners_cam[:, 0] * fx / corners_cam[:, 2]) + u0
                    proj_y = (corners_cam[:, 1] * fy / corners_cam[:, 2]) + v0
                    corners_2d = np.stack((proj_x, proj_y), axis=1).astype(int)
                    
                    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
                    for pt1, pt2 in edges:
                        cv2.line(debug_frames["3D Box Debug"], tuple(corners_2d[pt1]), tuple(corners_2d[pt2]), (0, 255, 255), 2)
                    
                    cv2.putText(debug_frames["3D Box Debug"], f"ID:{tid} {CLASS_NAMES.get(cls_id, 'unknown')} D:{Z:.1f}m", (corners_2d[0][0], corners_2d[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    print("3D Box\n↓")
                else:
                    print("Depth Pixels\nTotal Pixels : 0\n")
                    print("PIPELINE BROKEN AT:\nDepth Value\n")
            else:
                print("PIPELINE BROKEN AT:\nDepth Value\n")
            
            print("==================================")
            
            bx, by = None, None
            if homography_engine is not None:
                bev_pt = homography_engine.project_point(float(cx), float(cy))
                bx_raw, by_raw = bev_pt
                
                print(f"RAW_BEV=({bx_raw},{by_raw})")
                
                is_valid = True
                fail_reason = ""
                
                if np.isnan(bx_raw) or np.isnan(by_raw):
                    is_valid = False
                    fail_reason = "NAN"
                elif np.isinf(bx_raw) or np.isinf(by_raw):
                    is_valid = False
                    fail_reason = "INF"
                elif bx_raw < 0 or bx_raw > Cfg.BEV_WIDTH:
                    is_valid = False
                    fail_reason = "OUTSIDE_X"
                elif by_raw < 0 or by_raw > Cfg.BEV_HEIGHT:
                    is_valid = False
                    fail_reason = "OUTSIDE_Y"
                
                if is_valid:
                    inside_count += 1
                    
                    x_m = (bx_raw - Cfg.BEV_WIDTH / 2.0) * meters_per_pixel
                    y_m = (Cfg.BEV_HEIGHT - by_raw) * meters_per_pixel
                    
                    print("Validation=PASS")
                    print(f"FINAL_BEV=({bx_raw},{by_raw})")
                    print(f"X={x_m:.1f} m")
                    print(f"Y={y_m:.1f} m")
                    
                    if tid not in _track_states:
                        _track_states[tid] = {
                            'history': deque(maxlen=15),
                            'vx_ema': 0.0, 'vy_ema': 0.0,
                            'age': 0,
                        }

                    state = _track_states[tid]
                    state['age'] += 1

                    vx_ema = state['vx_ema']
                    vy_ema = state['vy_ema']
                    
                    if len(state['history']) > 0:
                        prev_t, prev_x, prev_y, prev_bx, prev_by = state['history'][0]
                        dt = len(state['history']) * (1.0 / 30.0) # Assume 30 FPS video time
                        
                        if dt > 0.001:
                            dx_m = x_m - prev_x
                            dy_m = y_m - prev_y
                            vx_inst = dx_m / dt
                            vy_inst = dy_m / dt
                            
                            dbev_x = bx_raw - prev_bx
                            dbev_y = by_raw - prev_by
                            
                            if state['age'] <= 15:
                                vx_ema = vx_inst
                                vy_ema = vy_inst
                            else:
                                vx_ema = 0.2 * vx_inst + 0.8 * state['vx_ema']
                                vy_ema = 0.2 * vy_inst + 0.8 * state['vy_ema']

                            state['vx_ema'] = vx_ema
                            state['vy_ema'] = vy_ema
                            
                            print(f"dX={dx_m:.2f} dY={dy_m:.2f}")
                            print(f"ΔBEV_X={dbev_x} ΔBEV_Y={dbev_y}")
                            
                    state['history'].append((current_time, x_m, y_m, bx_raw, by_raw))
                    
                    speed = np.sqrt(vx_ema**2 + vy_ema**2)
                    rel_speed = vy_ema
                    
                    ttc = None
                    status = "GREEN"
                    color = (0, 255, 0)
                    
                    if state['age'] >= 10 and y_m >= 1.0 and rel_speed < -0.5:
                        ttc = y_m / abs(rel_speed)
                        if ttc <= 3.0:
                            status = "RED"
                            color = (0, 0, 255)
                        elif ttc <= 5.0:
                            status = "YELLOW"
                            color = (0, 255, 255)
                            
                    print(f"Distance={y_m:.1f}m")
                    print(f"VX={vx_ema:.1f}m/s")
                    print(f"VY={vy_ema:.1f}m/s")
                    print(f"Speed={speed:.1f}m/s")
                    if ttc is not None:
                        print(f"TTC={ttc:.1f}s")
                    
                    bx, by = bx_raw, by_raw
                    
                    cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                    length_m, width_m, bgr_color = VEHICLE_PROPS.get(cls_name, VEHICLE_PROPS['unknown'])

                    # Build physically correct aligned footprint
                    fp = None
                    if meters_per_pixel > 0 and state['age'] >= Cfg.MIN_TRUST_AGE:
                        print("Footprint Source\nSTATIC\n")
                        print("Vehicle")
                        print(f"Length (meters) : {length_m:.2f}")
                        print(f"Width (meters) : {width_m:.2f}")
                        print(f"Pixel Length : {int(length_m / meters_per_pixel)}")
                        print(f"Pixel Width : {int(width_m / meters_per_pixel)}")
                        print("Source\nSTATIC\n")
                        print("Ground Footprint\n↓")
                        print("BEV\n↓")
                        
                        fp = _build_vehicle_footprint(
                            track_id=tid,
                            cls_id=cls_id,
                            confidence=float(box[4]) if len(box) > 4 else 1.0,
                            bev_cx=bx, bev_cy=by,
                            x_m=x_m, y_m=y_m,
                            meters_per_pixel=meters_per_pixel,
                            distance_m=y_m,
                            velocity_y=rel_speed,
                            ttc=ttc,
                            status=status,
                            lane=lane_side,
                            class_names=CLASS_NAMES,
                            vehicle_props=VEHICLE_PROPS,
                            bev_w=Cfg.BEV_WIDTH,
                            bev_h=Cfg.BEV_HEIGHT,
                        )
                        if fp is not None:
                            _draw_footprint_on_bev(bev_vis, fp)

                    print(f"Vehicle:\nID={tid}\nClass={cls_name}")
                    print(f"Footprint: L={length_m}m  W={width_m}m")
                    print(f"Center: X={x_m:.1f}m  Y={y_m:.1f}m\n")

                    decorated_box = list(box) + [lane_side, validity, y_m, rel_speed, ttc, status, tid, length_m, width_m, cls_name, fp]
                    decorated_boxes.append(decorated_box)
                    
                else:
                    outside_count += 1
                    validity = "INVALID"
                    print("Validation=FAIL")
                    print(f"Reason={fail_reason}")
                    print("FINAL_BEV=(None,None)")
                    
                    color = (0, 0, 255) # Red for rejected
                    if fail_reason not in ["NAN", "INF"]:
                        cv2.circle(bev_vis, (bx_raw, by_raw), 6, color, -1)
                        cv2.putText(bev_vis, f"ID={tid}", (bx_raw + 10, by_raw), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    decorated_boxes.append(list(box) + ['UNKNOWN', 'INVALID', None, None, None, None, tid, 4.5, 1.8, 'unknown', None])
            else:
                decorated_boxes.append(list(box) + ['UNKNOWN', 'INVALID', None, None, None, None, tid, 4.5, 1.8, 'unknown', None])
            
            print("")
            
            if lane_side in lane_counts and validity == "VALID":
                lane_counts[lane_side] += 1
                
        else:
            decorated_boxes.append(list(box) + ['UNKNOWN', 'INVALID', None, None, None, None, None, 4.5, 1.8, 'unknown', None])
            
    active_tids = {b[17] for b in decorated_boxes if len(b) > 17 and b[17] is not None}
    stale = [t for t in _track_states if t not in active_tids]
    for t in stale:
        del _track_states[t]
        
    lane_occupancy = {"LEFT": [], "CENTER": [], "RIGHT": []}
    for b in decorated_boxes:
        if len(b) > 17 and b[12] == "VALID":
            lane = b[11]
            dist = b[13]
            if lane in lane_occupancy and dist is not None:
                lane_occupancy[lane].append(b)
                
    for lane in lane_occupancy:
        lane_occupancy[lane].sort(key=lambda x: x[13]) # closest first
        
    PLANNING_HORIZON = 30.0

    # ── Metric occupancy grid (replaces 3×20 boolean array) ────────
    # Collect all valid footprints for rasterization
    valid_footprints = [
        b[-1] for b in decorated_boxes
        if len(b) > 20 and b[12] == 'VALID' and b[-1] is not None
    ]
    occ_grid_bin = _rasterize_footprints_to_grid(valid_footprints, meters_per_pixel)

    # Reconstruct the 3×20 boolean grid from the metric grid for the
    # existing BehaviorPlanner API (lane-column × distance-row cells).
    # Each lane column maps to a lateral slice of the metric grid.
    free_grid = np.zeros((3, 20), dtype=bool)
    cell_len_m = PLANNING_HORIZON / 20.0
    lane_to_col = {'LEFT': 0, 'CENTER': 1, 'RIGHT': 2}

    grid_total_cols = _occ_grid_cols
    # Ego occupies the centre third of the lateral grid
    col_left_start  = 0
    col_left_end    = grid_total_cols // 3
    col_center_start = grid_total_cols // 3
    col_center_end   = 2 * grid_total_cols // 3
    col_right_start  = 2 * grid_total_cols // 3
    col_right_end    = grid_total_cols

    lane_col_slices = {
        'LEFT':   (col_left_start,   col_left_end),
        'CENTER': (col_center_start, col_center_end),
        'RIGHT':  (col_right_start,  col_right_end),
    }
    rows_per_cell = max(1, int(cell_len_m / Cfg.GRID_CELL_M))
    occ_grid_rows_total = _occ_grid_rows

    for lane_name, (cs, ce) in lane_col_slices.items():
        col_idx = lane_to_col[lane_name]
        for cell_idx in range(20):
            # cell 0 = nearest to ego (0→1.5 m), cell 19 = farthest
            # Grid row 0 = farthest, so farthest row in occ_grid is index 0
            # Distance from ego front:  y_start = cell_idx * cell_len_m
            y_start = cell_idx * cell_len_m
            y_end   = y_start + cell_len_m
            # Convert metric Y to grid row indices (row 0 = GRID_RANGE_M ahead)
            row_start = max(0, int((Cfg.GRID_RANGE_M - y_end)   / Cfg.GRID_CELL_M))
            row_end   = min(occ_grid_rows_total - 1,
                            int((Cfg.GRID_RANGE_M - y_start) / Cfg.GRID_CELL_M))
            if row_end >= row_start:
                slice_region = occ_grid_bin[row_start:row_end + 1, cs:ce]
                if slice_region.any():
                    free_grid[col_idx][cell_idx] = True
                    
    print("Lane Assessment:")
    bev_y_offset = 20
    best_score = -1.0
    best_corridor = "CENTER"
    lane_scores = {}
    gap_detection_map = {}
    free_space_map = {}
    
    for i, lane in enumerate(["LEFT", "CENTER", "RIGHT"]):
        cars = [c for c in lane_occupancy[lane] if c[13] <= PLANNING_HORIZON]
        count = len(cars)
        
        nearest_dist = max(0.0, cars[0][13] - (cars[0][18] + 0.5)/2) if count > 0 else PLANNING_HORIZON
        
        largest_gap = 0.0
        if count == 0:
            largest_gap = PLANNING_HORIZON
        elif count == 1:
            largest_gap = max(0.0, PLANNING_HORIZON - (cars[0][13] + (cars[0][18] + 0.5)/2))
        else:
            gaps = []
            for j in range(1, count):
                gap = (cars[j][13] - (cars[j][18] + 0.5)/2) - (cars[j-1][13] + (cars[j-1][18] + 0.5)/2)
                gaps.append(max(0.0, gap))
            gaps.append(max(0.0, PLANNING_HORIZON - (cars[-1][13] + (cars[-1][18] + 0.5)/2)))
            largest_gap = max(gaps)

        free_cells = 20 - np.sum(free_grid[i])
        free_ratio = free_cells / 20.0
        
        score = 0.6 * free_ratio + 0.4 * (largest_gap / PLANNING_HORIZON)
        lane_scores[lane] = score
        gap_detection_map[lane] = largest_gap
        free_space_map[lane] = free_ratio
        
        if score > best_score:
            best_score = score
            best_corridor = lane
            
        print(f"{lane}:")
        print(f"  Nearest = {nearest_dist:.1f}m")
        print(f"  Largest Gap = {largest_gap:.1f}m")
        print(f"  Free Ratio = {free_ratio:.2f}")
        print(f"  Score = {score:.2f}")
        print("")
        
        state = "FREE"
        if nearest_dist < 10.0:
            state = "BLOCKED"
        elif nearest_dist < 20.0:
            state = "CAUTION"
            
        color = (0, 255, 0)
        if state == "BLOCKED":
            color = (0, 0, 255)
        elif state == "CAUTION":
            color = (0, 255, 255)
            
        cv2.putText(bev_vis, f"{lane}: {state}", (10, bev_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(bev_vis, f"Gap={largest_gap:.1f}m", (10, bev_y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(bev_vis, f"{int(free_ratio*100)}% FREE", (10, bev_y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        bev_y_offset += 45
        
    print(f"BEST CORRIDOR:\n{best_corridor}\n")
    cv2.putText(bev_vis, f"Best Corridor:", (10, Cfg.BEV_HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(bev_vis, f"{best_corridor}", (10, Cfg.BEV_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    lead_vehicle = lane_occupancy["CENTER"][0] if len(lane_occupancy["CENTER"]) > 0 else None
                
    decision, reason, confidence = _planner.plan(
        lane_occupancy, free_space_map, gap_detection_map, lane_scores, lead_vehicle
    )
    
    print("Behavior Planner")
    if lead_vehicle is not None:
        print(f"Lead Vehicle: ID={int(lead_vehicle[17])} Distance={lead_vehicle[13]:.1f}m")
    else:
        print("Lead Vehicle: None")
    print("Current Lane=CENTER")
    
    left_safe = len([c for c in lane_occupancy["LEFT"] if c[13] < 10.0]) == 0
    right_safe = len([c for c in lane_occupancy["RIGHT"] if c[13] < 10.0]) == 0
    print(f"Left Safe={left_safe}")
    print(f"Right Safe={right_safe}")
    print(f"Largest Left Gap={gap_detection_map.get('LEFT', 0.0):.0f}m")
    print(f"Largest Right Gap={gap_detection_map.get('RIGHT', 0.0):.0f}m")
    print(f"Decision={decision.value}")
    print(f"Confidence={confidence}%")
    print(f'Reason="{reason}"\n')
    
    # Visualization Panel
    cv2.rectangle(bev_vis, (200, Cfg.BEV_HEIGHT - 130), (390, Cfg.BEV_HEIGHT - 10), (30, 30, 30), -1)
    cv2.rectangle(bev_vis, (200, Cfg.BEV_HEIGHT - 130), (390, Cfg.BEV_HEIGHT - 10), (100, 100, 100), 1)
    cv2.putText(bev_vis, "BEHAVIOR PLANNER", (205, Cfg.BEV_HEIGHT - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(bev_vis, "State", (205, Cfg.BEV_HEIGHT - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(bev_vis, decision.value.replace("_", " "), (205, Cfg.BEV_HEIGHT - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.putText(bev_vis, "Reason", (205, Cfg.BEV_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(bev_vis, reason[:25], (205, Cfg.BEV_HEIGHT - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    if len(reason) > 25:
        cv2.putText(bev_vis, reason[25:50], (205, Cfg.BEV_HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
    cv2.putText(bev_vis, f"Confidence: {confidence}%", (205, Cfg.BEV_HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    if len(car_boxes) > 0:
        print(f"Inside: {inside_count}\nOutside: {outside_count}\n")
        print("Lane Summary:")
        print(f"LEFT={lane_counts['LEFT']} CENTER={lane_counts['CENTER']} RIGHT={lane_counts['RIGHT']}\n")

    # Debug frames
    hls_vis = build_hls_vis(clean_mask, car_boxes)

    result = draw_overlay(
        frame.copy(), left, right, roi_poly,
        calibrated=_calibrator.is_locked if Cfg.USE_AUTOCAL else True,
        car_boxes=decorated_boxes,
        car_masks=car_masks,
        src_pts=src_pts,
        lane_pts=lane_pts
    )

    if Cfg.DEBUG_STREAM_SCALE != 1.0:
        dw = max(1, int(w * Cfg.DEBUG_STREAM_SCALE))
        dh = max(1, int(h * Cfg.DEBUG_STREAM_SCALE))
        hls_vis = cv2.resize(hls_vis, (dw, dh), interpolation=cv2.INTER_LINEAR)
        roi_vis = cv2.resize(roi_vis, (dw, dh), interpolation=cv2.INTER_LINEAR)

    if Cfg.PRINT_BENCHMARK:
        print(f"[BENCHMARK] {(time.perf_counter()-_t0)*1000:.1f}ms")
        
    if "Ground Footprint Debug" not in debug_frames:
        gfd = np.zeros((Cfg.BEV_HEIGHT, Cfg.BEV_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(gfd, (0, 0), (Cfg.BEV_WIDTH, Cfg.BEV_HEIGHT), (0, 40, 0), -1)
        if src_pts is not None and H is not None:
            cv2.line(gfd, (int(p_tl[0]), int(p_tl[1])), (int(p_bl[0]), int(p_bl[1])), (255, 255, 255), 2)
            cv2.line(gfd, (int(p_tr[0]), int(p_tr[1])), (int(p_br[0]), int(p_br[1])), (255, 255, 255), 2)
        if meters_per_pixel > 0:
            cv2.rectangle(gfd, (ex - ego_wid_px//2, ey - ego_len_px//2),
                          (ex + ego_wid_px//2, ey + ego_len_px//2), (255, 255, 255), -1)
            cv2.putText(gfd, "EGO", (ex - 15, ey + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for box in decorated_boxes:
            fp = box[-1]
            if fp is not None:
                cv2.polylines(gfd, [fp.bev_polygon.reshape(-1, 1, 2)], True, fp.color, 2, cv2.LINE_AA)
                cv2.putText(gfd, f"ID{fp.track_id}", (int(fp.bev_polygon[:,0].min()), int(fp.bev_polygon[:,1].min()-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        debug_frames["Ground Footprint Debug"] = gfd

    return result, hls_vis, roi_vis, bev_vis, debug_frames


def process_video(source=0, show_debug=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return
    print("q=quit | d=debug")
    debug = show_debug
    while True:
        ret, frame = cap.read()
        if not ret: break
        result, hls_vis, roi_vis, bev_vis, debug_frames = detect_lanes(frame)
        cv2.imshow("Lane Detection", result)
        cv2.imshow("Bird Eye View", bev_vis)
        if debug:
            cv2.imshow("HLS + Cars", hls_vis)
            cv2.imshow("ROI Edges",  roi_vis)
            for k, v in debug_frames.items():
                cv2.imshow(k, v)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('d'): debug = not debug
    _detector.stop()
    cap.release()
    cv2.destroyAllWindows()


def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: {path}"); return
    result, hls_vis, roi_vis, bev_vis, debug_frames = detect_lanes(frame)
    cv2.imshow("Lane Detection", result)
    cv2.imshow("Bird Eye View", bev_vis)
    cv2.imshow("HLS + Cars",     hls_vis)
    cv2.imshow("ROI Edges",      roi_vis)
    for k, v in debug_frames.items():
        cv2.imshow(k, v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    src  = args[0] if args else 0
    if isinstance(src, str) and src.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")):
        process_image(src)
    else:
        process_video(source=src, show_debug="--debug" in args)