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


class Cfg:

    # ── ROI ────────────────────────────────────────────────────────
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

    # [P6] Debug stream scale (0.5 = half-res, saves ~75% encode time)
    DEBUG_STREAM_SCALE = 0.5

    # Benchmark — set True only for local profiling
    PRINT_BENCHMARK    = False


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

    # ── called from pipeline thread every frame ───────────────────
    def detect(self, frame):
        """
        Non-blocking: submits frame to worker every YOLO_EVERY_N frames,
        always returns the latest cached boxes immediately.
        """
        self._ensure_started()
        self._frame_idx += 1

        if not Cfg.USE_YOLO or not _YOLO_AVAILABLE:
            return [], []

        # Only submit a new frame if the worker is ready (queue not full)
        if self._frame_idx % 1 == 0:     # every frame — worker decides pace
            try:
                # drop_if_full: discard old frame, never block pipeline
                self._in_q.put_nowait(frame.copy())
            except queue.Full:
                pass    # worker still busy — use cached result

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [J] Car-based ROI top
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HLS lane mask  [P4] returns gray to avoid double conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Slope filter + cross clamp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Debug vis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Drawing  [P7] np.where blend instead of addWeighted + copy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 1.5 - Validation & Lane Assignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Module-level singletons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_smoother   = _Smoother(Cfg.SMOOTH_FRAMES)
_ema_cross  = _EMA(alpha=Cfg.EMA_ALPHA)
_calibrator = _AutoCalibrator()
_detector   = _AsyncCarDetector()          # [P5] async
_homography = _HomographyTracker()
_track_states = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 8 - Behavior Planner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    # [P5] Non-blocking: submit frame, get latest cached boxes
    car_boxes, car_masks = _detector.detect(frame)

    # [A] HLS mask — [P4] also returns gray
    masked_gray, raw_mask, gray = hls_lane_mask(frame)

    # [I] Subtract cars
    clean_mask  = mask_cars(raw_mask, car_boxes)
    masked_gray = cv2.bitwise_and(gray, clean_mask)

    # [B] Otsu Canny
    edges = otsu_canny(masked_gray)

    # [F][H][J] Dynamic ROI top
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

    # [G] Pentagon ROI
    roi_masked, roi_vis, roi_poly = apply_roi(edges, h, w, top_y_frac)

    # Hough
    lines = cv2.HoughLinesP(
        roi_masked,
        rho=Cfg.HOUGH_RHO, theta=Cfg.HOUGH_THETA,
        threshold=Cfg.HOUGH_THRESHOLD,
        minLineLength=Cfg.HOUGH_MIN_LENGTH,
        maxLineGap=Cfg.HOUGH_MAX_GAP,
    )

    # [C][D] Filter + fit
    left_raw, right_raw, cross_y = filter_and_fit(lines, h, w, top_y_frac)

    # [F] EMA update
    _ema_cross.update(cross_y)
    if Cfg.USE_AUTOCAL:
        _calibrator.update(cross_y, h)

    # [E] Smooth
    left, right = _smoother.update(left_raw, right_raw)

    # [L] BEV Homography
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
    
    if H is not None:
        det_H = np.linalg.det(H)
        print(f"det(H)={det_H:.6f}")
        if abs(det_H) < 1e-6:
            print("WARNING: DEGENERATE HOMOGRAPHY")
            
        pts_to_proj = np.array([[[pt[0], pt[1]]] for pt in lane_pts], dtype=np.float32)
        proj_lane_pts = cv2.perspectiveTransform(pts_to_proj, H)
        p_tl, p_tr, p_bl, p_br = proj_lane_pts[:, 0, :]
        lane_w_px_top = p_tr[0] - p_tl[0]
        lane_w_px_bot = p_br[0] - p_bl[0]
        lane_w_px = (lane_w_px_top + lane_w_px_bot) / 2.0
        
        meters_per_pixel = Cfg.LANE_WIDTH_M / max(1.0, lane_w_px)
        
        print(f"Lane Width BEV = {lane_w_px:.1f} px")
        print(f"Meters Per Pixel = {meters_per_pixel:.4f} m/px\n")
    else:
        meters_per_pixel = 0.0

    # Phase 7 - Drivable Corridor Fill
    bev_vis = np.zeros((Cfg.BEV_HEIGHT, Cfg.BEV_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(bev_vis, (0, 0), (Cfg.BEV_WIDTH, Cfg.BEV_HEIGHT), (0, 40, 0), -1)
    
    if src_pts is not None and H is not None:
        pts_to_project = np.array([[[pt[0], pt[1]]] for pt in lane_pts], dtype=np.float32)
        proj_lane_pts = cv2.perspectiveTransform(pts_to_project, H)
        p_tl, p_tr, p_bl, p_br = proj_lane_pts[:, 0, :]
        
        cv2.line(bev_vis, (int(p_tl[0]), int(p_tl[1])), (int(p_bl[0]), int(p_bl[1])), (0, 255, 0), 2)
        cv2.line(bev_vis, (int(p_tr[0]), int(p_tr[1])), (int(p_br[0]), int(p_br[1])), (0, 255, 0), 2)
        
        cx_top = int((p_tl[0] + p_tr[0]) / 2)
        cx_bot = int((p_bl[0] + p_br[0]) / 2)
        cv2.line(bev_vis, (cx_top, int(p_tl[1])), (cx_bot, int(p_bl[1])), (255, 255, 255), 1)

    # [K] Phase 1.5 Validation & Output
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
            
            print(f"ID={tid}")
            print(f"IMG=({cx},{cy})")
            
            bx, by = None, None
            if H is not None:
                pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                bev_pt = cv2.perspectiveTransform(pt, H)
                bx_raw, by_raw = int(bev_pt[0][0][0]), int(bev_pt[0][0][1])
                
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
                        _track_states[tid] = {'history': deque(maxlen=15), 'vx_ema': 0.0, 'vy_ema': 0.0, 'age': 0}
                        
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
                    
                    cls_name = CLASS_NAMES.get(cls_id, "unknown")
                    length_m, width_m, bgr_color = VEHICLE_PROPS.get(cls_name, VEHICLE_PROPS["unknown"])
                    
                    if meters_per_pixel > 0:
                        len_px = int(length_m / meters_per_pixel)
                        wid_px = int(width_m / meters_per_pixel)
                        
                        cv2.rectangle(bev_vis, (bx - wid_px//2, by - len_px//2),
                                      (bx + wid_px//2, by + len_px//2), bgr_color, -1)
                                      
                        cv2.putText(bev_vis, f"ID {tid}", (bx - wid_px//2, by - len_px//2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1)
                        cv2.putText(bev_vis, f"{y_m:.1f} m", (bx - wid_px//2, by - len_px//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    print(f"Vehicle:\nID={tid}\nClass={cls_name}\nFootprint:\nLength={length_m}m\nWidth={width_m}m")
                    print(f"Pixel:\nL={int(length_m / max(0.0001, meters_per_pixel))} px\nW={int(width_m / max(0.0001, meters_per_pixel))} px")
                    print(f"Center:\nX={x_m:.1f}m\nY={y_m:.1f}m\n")
                    
                    decorated_box = list(box) + [lane_side, validity, y_m, rel_speed, ttc, status, tid, length_m, width_m, cls_name]
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
                        
                    decorated_boxes.append(list(box) + ["UNKNOWN", "INVALID", None, None, None, None, tid, 4.5, 1.8, "unknown"])
            else:
                decorated_boxes.append(list(box) + ["UNKNOWN", "INVALID", None, None, None, None, tid, 4.5, 1.8, "unknown"])
            
            print("")
            
            if lane_side in lane_counts and validity == "VALID":
                lane_counts[lane_side] += 1
                
        else:
            decorated_boxes.append(list(box) + ["UNKNOWN", "INVALID", None, None, None, None, None, 4.5, 1.8, "unknown"])
            
    active_tids = {b[17] for b in decorated_boxes if len(b) > 17 and b[17] is not None}
    stale = [t for t in _track_states if t not in active_tids]
    for t in stale:
        del _track_states[t]
        
    # Phase 5 - Lane Occupancy Map
    lane_occupancy = {"LEFT": [], "CENTER": [], "RIGHT": []}
    for b in decorated_boxes:
        if len(b) > 17 and b[12] == "VALID":
            lane = b[11]
            dist = b[13]
            if lane in lane_occupancy and dist is not None:
                lane_occupancy[lane].append(b)
                
    for lane in lane_occupancy:
        lane_occupancy[lane].sort(key=lambda x: x[13]) # closest first
        
    # Phase 7.1 - Free Space Grid with 30m Planning Horizon
    PLANNING_HORIZON = 30.0
    free_grid = np.zeros((3, 20), dtype=bool)
    cell_len_m = PLANNING_HORIZON / 20.0
    lane_to_col = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
    
    for b in decorated_boxes:
        if len(b) > 17 and b[12] == "VALID":
            lane = b[11]
            y_m = b[13]
            length_m = b[18]
            if lane in lane_to_col and y_m is not None and y_m <= PLANNING_HORIZON:
                col = lane_to_col[lane]
                occ_len = length_m + 0.5
                start_row = int(max(0.0, y_m - occ_len/2) / cell_len_m)
                end_row = int(min(PLANNING_HORIZON - 0.001, y_m + occ_len/2) / cell_len_m)
                for r in range(start_row, min(20, end_row + 1)):
                    free_grid[col][r] = True
                    
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

    # [P6] Downscale debug frames — cheaper to encode & stream
    if Cfg.DEBUG_STREAM_SCALE != 1.0:
        dw = max(1, int(w * Cfg.DEBUG_STREAM_SCALE))
        dh = max(1, int(h * Cfg.DEBUG_STREAM_SCALE))
        hls_vis = cv2.resize(hls_vis, (dw, dh), interpolation=cv2.INTER_LINEAR)
        roi_vis = cv2.resize(roi_vis, (dw, dh), interpolation=cv2.INTER_LINEAR)

    if Cfg.PRINT_BENCHMARK:
        print(f"[BENCHMARK] {(time.perf_counter()-_t0)*1000:.1f}ms")

    return result, hls_vis, roi_vis, bev_vis


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
        result, hls_vis, roi_vis, bev_vis = detect_lanes(frame)
        cv2.imshow("Lane Detection", result)
        cv2.imshow("Bird Eye View", bev_vis)
        if debug:
            cv2.imshow("HLS + Cars", hls_vis)
            cv2.imshow("ROI Edges",  roi_vis)
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
    result, hls_vis, roi_vis, bev_vis = detect_lanes(frame)
    cv2.imshow("Lane Detection", result)
    cv2.imshow("Bird Eye View", bev_vis)
    cv2.imshow("HLS + Cars",     hls_vis)
    cv2.imshow("ROI Edges",      roi_vis)
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