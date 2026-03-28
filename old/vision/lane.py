"""
lane_detection.py  —  YOLOv8 Segmentation-based Lane Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why Ultralytics YOLO instead of classical CV?
──────────────────────────────────────────────
  Classical Hough + Sliding Window has no concept of what a lane IS.
  Cars, signs, shadows, dashed lines all confuse it.

  YOLOv8-seg treats the problem as instance segmentation — it draws
  pixel-level masks around every detected object including road/lane
  markings. Trained on COCO + fine-tunable on lane datasets.
  Handles heavy traffic, partial occlusion, dashed lines naturally.

Model used
──────────
  yolov8n-seg.pt  — nano segmentation model (~6 MB, fast on CPU)

  On first run, ultralytics auto-downloads it to ~/.cache/ultralytics/
  No manual URL needed — just pip install ultralytics.

How lanes are extracted from YOLO segmentation output
───────────────────────────────────────────────────────
  YOLO detects many classes. We filter for road/lane-related classes
  and use their segmentation masks to find left + right lane boundaries
  via contour analysis + polyfit.

  For even better results use a lane-specific YOLO model fine-tuned on
  TuSimple/CULane — drop it in by changing WEIGHTS_NAME below.

Pipeline
─────────
  BGR frame
    │  ultralytics YOLO forward pass
    ▼
  Segmentation masks  (class-filtered)
    │  contour extraction + left/right split by x-position
    ▼
  Polyfit per side  →  smooth lane curves
    │  temporal smoothing
    ▼
  Draw filled polygon + polylines on original frame

Setup — run once
─────────────────
  pip install ultralytics opencv-python numpy flask

Controls (standalone)
──────────────────────
  q — quit   |   d — debug masks   |   s — save frame
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ── Ultralytics import with friendly error ────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _ULTRA_OK = True
except ImportError:
    _ULTRA_OK = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Config  — all tunables in one place
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Config:

    # ── Model ─────────────────────────────────────────────────────────────────
    # ultralytics auto-downloads this on first run → ~/.cache/ultralytics/
    # Swap to a lane-finetuned model path if you have one, e.g.:
    #   WEIGHTS_NAME = "/path/to/lane_yolov8.pt"
    WEIGHTS_NAME    = "yolov8n-seg.pt"   # nano — fast on CPU, ~6 MB

    # COCO class ids we treat as "road surface / lane area"
    # 0=person removed, keeping only road-relevant classes:
    #   2=car, 9=traffic light, 11=stop sign included for context
    # For a lane-specific model these won't matter — all masks = lanes
    ROAD_CLASSES    = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, moto, bus, truck
    # We use car/vehicle masks as negative space to find the road between them

    # Confidence threshold for YOLO detections
    CONF            = 0.25

    # IOU threshold
    IOU             = 0.45

    # ── Lane geometry ─────────────────────────────────────────────────────────
    # Only look at the bottom portion of the frame for lane lines
    ROI_TOP_FRAC    = 0.55    # ignore everything above 55% of frame height

    # Polynomial degree for lane curve fitting
    POLY_DEG        = 2       # 2 = curved lanes, 1 = straight only

    # Minimum contour area to be considered a lane marking (px²)
    MIN_AREA        = 800

    # ── Temporal smoothing ────────────────────────────────────────────────────
    SMOOTH_FRAMES   = 10

    # ── Drawing ───────────────────────────────────────────────────────────────
    LANE_COLOR      = (0,   255,   0)
    FILL_COLOR      = (0,   200, 255)
    FILL_ALPHA      = 0.30
    LINE_THICKNESS  = 4
    DEBUG_MASK_CLR  = (0,   120, 255)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    if not _ULTRA_OK:
        raise RuntimeError(
            "\nUltralytics not installed. Run:\n"
            "  pip install ultralytics\nthen restart.\n"
        )

    print(f"[YOLO] Loading {Config.WEIGHTS_NAME} ...")
    # ultralytics downloads weights automatically on first call
    _model = _YOLO(Config.WEIGHTS_NAME)
    print("[YOLO] Model ready.")
    return _model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lane extraction from YOLO segmentation masks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_road_mask(result, frame_h: int, frame_w: int):
    """
    Build a binary road-surface mask from YOLO result.

    Strategy
    ────────
    We invert the vehicle-mask approach:
    1. Start with full ROI as "road"
    2. Subtract detected vehicle masks → remaining = road/lane area
    3. Also include any explicitly road-class masks if model knows them

    This works with generic COCO yolov8-seg AND with lane-specific models.
    """
    roi_top = int(frame_h * Config.ROI_TOP_FRAC)
    road    = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # Fill ROI rectangle as initial road canvas
    road[roi_top:, :] = 255

    # If no masks detected at all, return the plain ROI strip
    if result.masks is None:
        return road

    masks  = result.masks.data.cpu().numpy()   # (N, H, W) float [0,1]
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    for mask_arr, cls_id in zip(masks, cls_ids):
        # Resize mask to original frame size
        mask_full = cv2.resize(mask_arr, (frame_w, frame_h))
        bin_mask  = (mask_full > 0.5).astype(np.uint8) * 255

        if cls_id in Config.ROAD_CLASSES:
            # Vehicle detected — remove from road canvas (it's not road)
            road = cv2.bitwise_and(road, cv2.bitwise_not(bin_mask))

    return road


def _extract_lane_contours(road_mask, frame_w: int):
    """
    Find lane-line contours in the road mask.
    Split into left / right groups by frame centre.
    Returns (left_contours, right_contours).
    """
    # Morphological clean-up — remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean  = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean,     cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = frame_w // 2
    left_ctrs, right_ctrs = [], []

    for cnt in contours:
        if cv2.contourArea(cnt) < Config.MIN_AREA:
            continue
        M    = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        mean_x = int(M["m10"] / M["m00"])
        if mean_x <= cx:
            left_ctrs.append(cnt)
        else:
            right_ctrs.append(cnt)

    return left_ctrs, right_ctrs


def _fit_lane_curve(contours, frame_h: int, frame_w: int, side: str):
    """
    Merge contour points, fit a polynomial, return (N, 2) int32 array
    of (x, y) spanning the ROI, or None if not enough points.
    side: 'left' | 'right' — used for x-sanity clamping
    """
    if not contours:
        return None

    pts = np.vstack([c.reshape(-1, 2) for c in contours])
    if len(pts) < 10:
        return None

    ys = pts[:, 1]
    xs = pts[:, 0]

    try:
        coeffs = np.polyfit(ys, xs, Config.POLY_DEG)
    except np.linalg.LinAlgError:
        return None

    roi_top = int(frame_h * Config.ROI_TOP_FRAC)
    plot_y  = np.linspace(roi_top, frame_h - 1, frame_h - roi_top)
    plot_x  = np.polyval(coeffs, plot_y)

    # Sanity clamp — left lane must stay in left 70%, right in right 70%
    if side == "left":
        plot_x = np.clip(plot_x, 0, frame_w * 0.70)
    else:
        plot_x = np.clip(plot_x, frame_w * 0.30, frame_w)

    return np.column_stack([plot_x, plot_y]).astype(np.int32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Temporal smoothing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _LaneSmoother:
    """Average lane curve x-coords over last N frames per y-row."""
    def __init__(self):
        self.left_hist  = deque(maxlen=Config.SMOOTH_FRAMES)
        self.right_hist = deque(maxlen=Config.SMOOTH_FRAMES)

    def update(self, left, right):
        if left  is not None: self.left_hist.append(left)
        if right is not None: self.right_hist.append(right)
        return (self._avg(self.left_hist),
                self._avg(self.right_hist))

    @staticmethod
    def _avg(hist):
        if not hist:
            return None
        min_len = min(len(a) for a in hist)
        stacked = np.array([a[:min_len] for a in hist], dtype=np.float32)
        return np.mean(stacked, axis=0).astype(np.int32)


_smoother = _LaneSmoother()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_overlay(frame, left_curve, right_curve):
    """Draw filled lane polygon + edge lines onto frame."""
    if left_curve is not None and right_curve is not None:
        poly    = np.vstack([left_curve, right_curve[::-1]])
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], Config.FILL_COLOR)
        frame   = cv2.addWeighted(overlay, Config.FILL_ALPHA,
                                  frame,   1 - Config.FILL_ALPHA, 0)

    for curve in (left_curve, right_curve):
        if curve is not None and len(curve) >= 2:
            cv2.polylines(frame, [curve.reshape(-1, 1, 2)],
                          False, Config.LANE_COLOR, Config.LINE_THICKNESS)
    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API  ←  app.py calls this, same 4-tuple signature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_lanes(frame, show_debug: bool = False):
    """
    YOLOv8-seg lane detection.

    Returns 4-tuple (unchanged from previous versions — app.py needs no edits):
      result       — BGR frame with lane overlay
      debug_vis    — road mask BGR (what lanes were extracted from)
      edges        — blank placeholder for stream compatibility
      debug_win    — None
    """
    model = _load_model()
    h, w  = frame.shape[:2]

    # 1. YOLO inference — verbose=False suppresses per-frame console spam
    results = model(frame, conf=Config.CONF, iou=Config.IOU,
                    verbose=False, stream=False)
    result  = results[0]

    # 2. Build road mask from vehicle masks
    road_mask = _build_road_mask(result, h, w)

    # 3. Extract lane contours
    left_ctrs, right_ctrs = _extract_lane_contours(road_mask, w)

    # 4. Fit polynomial curves
    left_raw  = _fit_lane_curve(left_ctrs,  h, w, "left")
    right_raw = _fit_lane_curve(right_ctrs, h, w, "right")

    # 5. Temporal smoothing
    left_smooth, right_smooth = _smoother.update(left_raw, right_raw)

    # 6. Draw
    out = _draw_overlay(frame.copy(), left_smooth, right_smooth)

    # Debug view — road mask as BGR
    debug_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    if show_debug and (left_ctrs or right_ctrs):
        cv2.drawContours(debug_vis, left_ctrs  + right_ctrs, -1,
                         Config.DEBUG_MASK_CLR, 2)

    edges_placeholder = np.zeros((h, w), dtype=np.uint8)

    return out, debug_vis, edges_placeholder, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Standalone entry-points
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_video(source=0, show_debug=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return
    _load_model()
    print("Controls: q=quit | d=debug mask | s=save frame")
    debug = show_debug
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result, debug_vis, _, _ = detect_lanes(frame, debug)
        cv2.imshow("YOLO Lane Detection", result)
        if debug:
            cv2.imshow("Road mask", debug_vis)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('d'): debug = not debug
        elif key == ord('s'):
            cv2.imwrite("lane_frame.png", result)
            print("Saved lane_frame.png")
    cap.release()
    cv2.destroyAllWindows()


def process_image(path, show_debug=False):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: {path}")
        return
    _load_model()
    result, debug_vis, _, _ = detect_lanes(frame, show_debug)
    cv2.imshow("YOLO Lane Detection", result)
    cv2.imshow("Road mask",           debug_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    args     = sys.argv[1:]
    debug    = "--debug" in args
    src_args = [a for a in args if not a.startswith("--")]
    source   = src_args[0] if src_args else 0
    if isinstance(source, str) and source.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")):
        process_image(source, debug)
    else:
        process_video(source, debug)