"""
lane_detection.py — Week 3 Research Build
==========================================
Week 1: [A] HLS mask  [B] Otsu Canny  [C] Cross clamp  [D] RANSAC  [E] Smoother
Week 2: [F] EMA ROI   [G] Pentagon    [H] Auto-calibration
Week 3: [I] YOLO car detection → mask cars out of HLS before Hough
        [J] Car-based dynamic ROI top
        [K] YOLO every Nth frame (speed optimisation)

detect_lanes() returns:
  result   — BGR overlay          → app.py "result"
  hls_vis  — BGR HLS+car debug    → app.py "edges"
  roi_vis  — BGR ROI edges debug  → app.py "roi"
"""

import cv2
import numpy as np
from collections import deque

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Cfg:

    # ── ROI ────────────────────────────────────────────────────────
    ROI_BOTTOM_LEFT_X  = 0.10
    ROI_BOTTOM_RIGHT_X = 0.90
    ROI_TOP_LEFT_X     = 0.43
    ROI_TOP_RIGHT_X    = 0.57
    ROI_TOP_Y          = 0.65   # fallback — overridden by [F][J]

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
    YOLO_MODEL         = "yolov8n.pt"   # auto-downloaded first run
    YOLO_CONF          = 0.35           # lower = detect more cars
    VEHICLE_CLASSES    = [2, 3, 5, 7]   # car, motorbike, bus, truck
    MASK_METHOD        = "confidence"        # direct | erode | confidence
    CAR_BOX_PADDING    = 10             # px expand before masking

    # ── [K] YOLO every N frames ────────────────────────────────────
    YOLO_EVERY_N       = 5             # run YOLO every 3rd frame

    # ── [J] Car-based ROI top ──────────────────────────────────────
    USE_CAR_ROI        = True          # use nearest center car bottom as ROI top
    CAR_ROI_BUFFER_PX  = 25            # px below car bottom before ROI starts
    CAR_CENTER_MIN_X   = 0.20          # car must be in center zone of frame
    CAR_CENTER_MAX_X   = 0.80

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
    CAR_BOX_COLOR      = (0,  60, 220)   # red boxes on debug view


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [I][K] YOLO detector — singleton, runs every Nth frame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _CarDetector:
    def __init__(self):
        self._model      = None
        self._last_boxes = []   # cached boxes from last YOLO run
        self._frame_idx  = 0

    def _load(self):
        if self._model is None and _YOLO_AVAILABLE and Cfg.USE_YOLO:
            print("[YOLO] Loading model…")
            self._model = _YOLO(Cfg.YOLO_MODEL)
            print("[YOLO] Ready")

    def detect(self, frame):
        """
        Returns list of [x1,y1,x2,y2,conf] arrays.
        Runs YOLO only every YOLO_EVERY_N frames — returns cached otherwise.
        """
        self._frame_idx += 1

        if not Cfg.USE_YOLO or not _YOLO_AVAILABLE:
            return []

        self._load()

        if self._frame_idx % Cfg.YOLO_EVERY_N != 0:
            return self._last_boxes   # use cached

        results = self._model(
            frame,
            verbose=False,
            conf=Cfg.YOLO_CONF,
            classes=Cfg.VEHICLE_CLASSES,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._last_boxes = []
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        self._last_boxes = np.hstack([xyxy, conf]).tolist()
        return self._last_boxes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [I] Car masking — subtract car boxes from HLS mask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mask_cars(lane_mask, car_boxes):
    """
    Zero out car bounding box regions from HLS lane mask.
    method = 'direct'     — blackout entire box
    method = 'erode'      — shrink box 10% each side (protect adjacent lanes)
    method = 'confidence' — full blackout only if conf > 0.7
    """
    if not car_boxes:
        return lane_mask

    result = lane_mask.copy()
    pad    = Cfg.CAR_BOX_PADDING
    h, w   = lane_mask.shape[:2]

    for box in car_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]) if len(box) > 4 else 1.0

        # Expand box by padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if Cfg.MASK_METHOD == "direct":
            result[y1:y2, x1:x2] = 0

        elif Cfg.MASK_METHOD == "erode":
            bw = x2 - x1
            bh = y2 - y1
            ex = int(bw * 0.08)
            ey = int(bh * 0.08)
            result[y1+ey : y2-ey, x1+ex : x2-ex] = 0

        elif Cfg.MASK_METHOD == "confidence":
            if conf >= 0.70:
                result[y1:y2, x1:x2] = 0
            elif conf >= 0.40:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                hw = (x2 - x1) // 4
                hh = (y2 - y1) // 4
                result[cy-hh:cy+hh, cx-hw:cx+hw] = 0

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [J] Car-based ROI top
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def car_based_roi_top(car_boxes, frame_h, frame_w):
    """
    Find nearest car in ego lane (center zone).
    Returns roi_top fraction or None if no center car found.
    """
    if not Cfg.USE_CAR_ROI or not car_boxes:
        return None

    min_x = frame_w * Cfg.CAR_CENTER_MIN_X
    max_x = frame_w * Cfg.CAR_CENTER_MAX_X
    bottoms = []

    for box in car_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        car_cx = (x1 + x2) / 2
        if min_x < car_cx < max_x:
            bottoms.append(y2)

    if not bottoms:
        return None

    nearest_bottom = max(bottoms)   # lowest bottom = nearest car
    roi_y = (nearest_bottom + Cfg.CAR_ROI_BUFFER_PX) / frame_h
    return float(np.clip(roi_y, Cfg.ROI_TOP_MIN, Cfg.ROI_TOP_MAX))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HLS lane mask (Week 1 — unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hls_lane_mask(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H   = hls[:, :, 0].astype(np.int32)
    L   = hls[:, :, 1].astype(np.int32)
    S   = hls[:, :, 2].astype(np.int32)

    white_mask  = (L >= Cfg.WHITE_L_MIN)
    yellow_mask = ((H >= Cfg.YELLOW_H_MIN) &
                   (H <= Cfg.YELLOW_H_MAX) &
                   (S >= Cfg.YELLOW_S_MIN))

    sky_cut = np.ones_like(L, dtype=np.uint8)
    sky_cut[:int(L.shape[0] * Cfg.SKY_CUT_FRAC), :] = 0

    lane_mask = np.clip(
        white_mask.astype(np.uint8) + yellow_mask.astype(np.uint8), 0, 1
    ).astype(np.uint8)
    lane_mask = (lane_mask * sky_cut * 255).astype(np.uint8)

    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, lane_mask)

    return masked_gray, lane_mask   # return raw mask too for car masking


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Otsu Canny (Week 1 — unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def otsu_canny(gray_image):
    blurred  = cv2.GaussianBlur(gray_image, Cfg.BLUR_KERNEL, 0)
    otsu_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_low  = max(10, int(otsu_val * 0.5))
    t_high = max(20, int(otsu_val))
    return cv2.Canny(blurred, t_low, t_high)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMA (Week 2 — unchanged)
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
#  Auto-calibrator (Week 2 — unchanged)
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
#  Pentagon / trapezoid ROI builder (Week 2 — unchanged)
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
    else:
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
#  RANSAC fit (Week 1 — unchanged)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Slope filter + cross clamp (Week 1/2 — updated: returns cross_y)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filter_and_fit(lines, h, w, roi_top_frac):
    ls, li, rs, ri = [], [], [], []

    if lines is None:
        return None, None, None

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if x2 == x1:
            continue
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
#  Temporal smoother (Week 1 — unchanged)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Debug visualisation — car boxes drawn on HLS mask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_hls_vis(lane_mask, car_boxes):
    """BGR debug frame: white = lane pixels, red boxes = detected cars."""
    vis = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    for box in car_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = float(box[4]) if len(box) > 4 else 1.0
        cv2.rectangle(vis, (x1, y1), (x2, y2), Cfg.CAR_BOX_COLOR, 2)
        cv2.putText(vis, f"{conf:.2f}", (x1 + 3, y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, Cfg.CAR_BOX_COLOR, 1)
    return vis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_overlay(frame, left, right, roi_polygon, calibrated, n_cars):
    out = frame.copy()

    if left is not None and right is not None:
        pts = np.array([
            [left[0],  left[1]],
            [left[2],  left[3]],
            [right[2], right[3]],
            [right[0], right[1]],
        ], dtype=np.int32)
        ov = out.copy()
        cv2.fillPoly(ov, [pts], Cfg.FILL_COLOR)
        out = cv2.addWeighted(ov, Cfg.FILL_ALPHA, out, 1 - Cfg.FILL_ALPHA, 0)

    for ln in (left, right):
        if ln is not None:
            cv2.line(out, (ln[0], ln[1]), (ln[2], ln[3]),
                     Cfg.LANE_COLOR, Cfg.LINE_THICKNESS)

    roi_color = (0, 200, 0) if calibrated else (0, 180, 255)
    cv2.polylines(out, roi_polygon, True, roi_color, 1)

    status = f"CAL | cars={n_cars}" if calibrated else f"WARM | cars={n_cars}"
    cv2.putText(out, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Module-level state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_smoother   = _Smoother(Cfg.SMOOTH_FRAMES)
_ema_cross  = _EMA(alpha=Cfg.EMA_ALPHA)
_calibrator = _AutoCalibrator()
_detector   = _CarDetector()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_lanes(frame):
    """
    Returns
    -------
    result   : BGR — lane overlay frame
    hls_vis  : BGR — HLS mask with red car boxes  (→ "edges" stream)
    roi_vis  : BGR — ROI-masked Canny edges        (→ "roi"   stream)
    """
    h, w = frame.shape[:2]

    # ── [I][K] YOLO car detection ─────────────────────────────────
    car_boxes = _detector.detect(frame)

    # ── [A] HLS mask ──────────────────────────────────────────────
    masked_gray, raw_mask = hls_lane_mask(frame)

    # ── [I] Subtract car boxes from HLS mask ──────────────────────
    clean_mask  = mask_cars(raw_mask, car_boxes)
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, clean_mask)

    # ── [B] Otsu Canny ────────────────────────────────────────────
    edges = otsu_canny(masked_gray)

    # ── [F][H][J] Compute dynamic ROI top ─────────────────────────
    car_top = car_based_roi_top(car_boxes, h, w)     # [J]

    if Cfg.USE_AUTOCAL:
        cal_top = _calibrator.update(_ema_cross.value, h)  # [H]
    else:
        ema_val = _ema_cross.value
        if ema_val is not None:
            cal_top = float(np.clip(
                (ema_val + Cfg.CROSS_BUFFER_PX) / h,
                Cfg.ROI_TOP_MIN, Cfg.ROI_TOP_MAX))
        else:
            cal_top = Cfg.ROI_TOP_Y

    # Use whichever is more conservative (lower in frame = larger fraction)
    if car_top is not None:
        top_y_frac = max(cal_top, car_top)
    else:
        top_y_frac = cal_top

    # ── [G] Pentagon ROI ──────────────────────────────────────────
    roi_masked, roi_vis, roi_poly = apply_roi(edges, h, w, top_y_frac)

    # ── Hough ─────────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        roi_masked,
        rho           = Cfg.HOUGH_RHO,
        theta         = Cfg.HOUGH_THETA,
        threshold     = Cfg.HOUGH_THRESHOLD,
        minLineLength = Cfg.HOUGH_MIN_LENGTH,
        maxLineGap    = Cfg.HOUGH_MAX_GAP,
    )

    # ── [C][D] Filter + cross clamp ───────────────────────────────
    left_raw, right_raw, cross_y = filter_and_fit(lines, h, w, top_y_frac)

    # ── [F] Update EMA ────────────────────────────────────────────
    _ema_cross.update(cross_y)
    if Cfg.USE_AUTOCAL:
        _calibrator.update(cross_y, h)

    # ── [E] Smooth ────────────────────────────────────────────────
    left, right = _smoother.update(left_raw, right_raw)

    # ── Build debug HLS vis with car boxes ────────────────────────
    hls_vis = build_hls_vis(clean_mask, car_boxes)

    result = draw_overlay(
        frame.copy(), left, right, roi_poly,
        calibrated = _calibrator.is_locked if Cfg.USE_AUTOCAL else True,
        n_cars     = len(car_boxes),
    )

    return result, hls_vis, roi_vis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Standalone runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_video(source=0, show_debug=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return
    print("q=quit | d=debug")
    debug = show_debug
    while True:
        ret, frame = cap.read()
        if not ret: break
        result, hls_vis, roi_vis = detect_lanes(frame)
        cv2.imshow("Lane Detection — Week 3", result)
        if debug:
            cv2.imshow("HLS + Cars", hls_vis)
            cv2.imshow("ROI Edges",  roi_vis)
        else:
            for win in ("HLS + Cars", "ROI Edges"):
                try: cv2.destroyWindow(win)
                except: pass
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('d'): debug = not debug
    cap.release()
    cv2.destroyAllWindows()


def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: {path}"); return
    result, hls_vis, roi_vis = detect_lanes(frame)
    cv2.imshow("Lane Detection — Week 3", result)
    cv2.imshow("HLS + Cars",              hls_vis)
    cv2.imshow("ROI Edges",               roi_vis)
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