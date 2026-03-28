"""
lane_detection.py — Week 2 Research Build
==========================================
Week 1 (carried forward — unchanged):
  [A] HLS color mask     — white + yellow pixels only
  [B] Otsu adaptive Canny — per-frame adaptive thresholds
  [C] Cross-point clamp  — X cross geometrically impossible
  [D] RANSAC slope filter — outlier car edges auto-rejected
  [E] Temporal smoother  — deque N-frame average

Week 2 (new):
  [F] Dynamic ROI        — EMA-tracked cross_y drives ROI top automatically
  [G] Pentagon ROI       — V-notch at center-bottom excludes car body
  [H] Auto-calibration   — first 30 frames median sets ROI, no manual tuning

detect_lanes() returns:
  result   — BGR overlay frame          ← app.py "result" stream
  hls_vis  — BGR HLS lane mask debug    ← app.py "edges" stream
  roi_vis  — BGR ROI edges debug        ← app.py "roi"   stream
ALL THREE are BGR — no cvtColor needed in app.py.
"""

import cv2
import numpy as np
from collections import deque

try:
    from sklearn.linear_model import RANSACRegressor
    _RANSAC_AVAILABLE = True
except ImportError:
    _RANSAC_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG — sirf yahan tune karo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Cfg:

    # ── ROI base shape ─────────────────────────────────────────────
    ROI_BOTTOM_LEFT_X  = 0.10
    ROI_BOTTOM_RIGHT_X = 0.90
    ROI_TOP_LEFT_X     = 0.43
    ROI_TOP_RIGHT_X    = 0.57
    ROI_TOP_Y          = 0.79   # fallback only — [F] overrides this dynamically

    # ── [F] Dynamic ROI — EMA cross tracking ──────────────────────
    EMA_ALPHA          = 0.15   # 0.10=very slow smooth, 0.30=faster response
    CROSS_BUFFER_PX    = 50     # ROI top = cross_y + this buffer (px)
    ROI_TOP_MIN        = 0.60   # ROI top never higher than 60% (safety)
    ROI_TOP_MAX        = 0.85   # ROI top never lower than 85%

    # ── [G] Pentagon ROI — car notch ──────────────────────────────
    USE_PENTAGON       = True   # False = trapezoid (Week 1 shape)
    NOTCH_TIP_Y        = 0.88   # V-tip y position (fraction of height)
    NOTCH_HALF_W       = 0.15   # notch half-width (fraction of width)
                                # increase if car edges still detected

    # ── [H] Auto-calibration ──────────────────────────────────────
    USE_AUTOCAL        = True   # False = use ROI_TOP_Y static value
    AUTOCAL_FRAMES     = 30     # warmup frames before calibration locks in
    AUTOCAL_BUFFER_PX  = 60     # extra buffer on top of median cross_y

    # ── HLS color mask (Week 1 — unchanged) ───────────────────────
    WHITE_L_MIN        = 110
    SKY_CUT_FRAC       = 0.55
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

    # ── Slope + x-position filter ──────────────────────────────────
    LEFT_SLOPE_MIN     = -0.9
    LEFT_SLOPE_MAX     = -0.4
    RIGHT_SLOPE_MIN    =  0.4
    RIGHT_SLOPE_MAX    =  0.9
    LEFT_X_MAX_FRAC    = 0.65
    RIGHT_X_MIN_FRAC   = 0.35

    # ── Cross-point draw clamp (Week 1 — unchanged) ────────────────
    CROSS_DRAW_BUFFER  = 30

    # ── Temporal smoother ──────────────────────────────────────────
    SMOOTH_FRAMES      = 7

    # ── Drawing ────────────────────────────────────────────────────
    LANE_COLOR         = (0, 255,   0)
    FILL_COLOR         = (0, 200, 255)
    FILL_ALPHA         = 0.30
    LINE_THICKNESS     = 6
    ROI_BORDER_COLOR   = (0, 255, 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 1 — [A] HLS Lane Mask (unchanged)
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
    hls_vis     = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    return masked_gray, hls_vis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 1 — [B] Otsu Adaptive Canny (unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def otsu_canny(gray_image):
    blurred  = cv2.GaussianBlur(gray_image, Cfg.BLUR_KERNEL, 0)
    otsu_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_low  = max(10, int(otsu_val * 0.5))
    t_high = max(20, int(otsu_val))
    return cv2.Canny(blurred, t_low, t_high)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 2 — [F] EMA cross_y tracker
#
#  Formula: ema = alpha * new_val + (1-alpha) * prev_ema
#  Why:     Recent frames matter more, old spikes fade out.
#           alpha=0.15 → ~6 frames lag (stable)
#           alpha=0.30 → ~3 frames lag (responsive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _EMA:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None   # None until first valid reading

    def update(self, new_val):
        if new_val is None:
            return self.value           # no measurement → hold last
        if self.value is None:
            self.value = float(new_val) # first reading → initialize
        else:
            self.value = (self.alpha * new_val +
                         (1.0 - self.alpha) * self.value)
        return self.value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 2 — [H] Auto-calibrator
#
#  Phase 1 (warmup): collect cross_y samples for N frames
#  Phase 2 (locked): roi_top = median(samples) + buffer — never changes
#  Why median not mean: 1 car-too-close frame won't spike the result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _AutoCalibrator:
    def __init__(self):
        self._samples  = []
        self._locked   = False
        self._roi_top  = Cfg.ROI_TOP_Y   # safe default during warmup

    @property
    def roi_top(self):
        return self._roi_top

    @property
    def is_locked(self):
        return self._locked

    def update(self, cross_y, frame_h):
        """Call every frame with current cross_y (can be None)."""
        if self._locked:
            return self._roi_top

        if cross_y is not None:
            self._samples.append(cross_y)

        if len(self._samples) >= Cfg.AUTOCAL_FRAMES:
            median_cross   = float(np.median(self._samples))
            safe_y         = median_cross + Cfg.AUTOCAL_BUFFER_PX
            self._roi_top  = float(np.clip(safe_y / frame_h,
                                           Cfg.ROI_TOP_MIN,
                                           Cfg.ROI_TOP_MAX))
            self._locked   = True
            print(f"[AutoCal] Locked — roi_top = {self._roi_top:.3f}  "
                  f"(median cross_y={median_cross:.0f}px, "
                  f"buffer={Cfg.AUTOCAL_BUFFER_PX}px)")

        return self._roi_top


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 2 — [G] Pentagon ROI builder
#
#  Shape: trapezoid + V-notch at center-bottom
#  Notch cuts out car silhouette zone
#  Lane markings live LEFT and RIGHT of the notch — unaffected
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_roi_polygon(h, w, top_y_frac):
    top_y   = int(h * top_y_frac)
    cx      = w // 2
    tip_y   = int(h * Cfg.NOTCH_TIP_Y)
    half_nw = int(w * Cfg.NOTCH_HALF_W)

    if Cfg.USE_PENTAGON:
        # 7-point pentagon with car notch
        polygon = np.array([[
            (int(w * Cfg.ROI_BOTTOM_LEFT_X),  h    ),  # 1 bottom-left
            (int(w * Cfg.ROI_TOP_LEFT_X),     top_y),  # 2 top-left
            (int(w * Cfg.ROI_TOP_RIGHT_X),    top_y),  # 3 top-right
            (int(w * Cfg.ROI_BOTTOM_RIGHT_X), h    ),  # 4 bottom-right
            (cx + half_nw,                    h    ),  # 5 notch-right
            (cx,                              tip_y),  # 6 notch-tip  (V)
            (cx - half_nw,                    h    ),  # 7 notch-left
        ]], dtype=np.int32)
    else:
        # Fallback: simple trapezoid (Week 1 shape)
        polygon = np.array([[
            (int(w * Cfg.ROI_BOTTOM_LEFT_X),  h    ),
            (int(w * Cfg.ROI_TOP_LEFT_X),     top_y),
            (int(w * Cfg.ROI_TOP_RIGHT_X),    top_y),
            (int(w * Cfg.ROI_BOTTOM_RIGHT_X), h    ),
        ]], dtype=np.int32)

    return polygon


def apply_roi(edges, h, w, top_y_frac):
    polygon = build_roi_polygon(h, w, top_y_frac)
    mask    = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked  = cv2.bitwise_and(edges, mask)
    roi_vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    return masked, roi_vis, polygon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 1 — [C]+[D] Slope filter + cross clamp (updated: returns cross_y)
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


def filter_and_fit(lines, h, w, roi_top_frac):
    """
    Returns (left_line, right_line, cross_y)
    cross_y is the pixel y where both lane lines would meet — fed to EMA.
    """
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

    # Compute cross_y — used by EMA + AutoCal
    cross_y = None
    if lm and rm and abs(lm - rm) > 1e-6:
        xc      = (rb - lb) / (lm - rm)
        cross_y = int(lm * xc + lb)

    def coords(m, b):
        if m is None:
            return None
        y1c = h
        y2c = int(h * roi_top_frac)
        # [C] clamp draw top to cross_y + draw buffer
        if cross_y is not None:
            y2c = max(y2c, cross_y + Cfg.CROSS_DRAW_BUFFER)
        x1c = int((y1c - b) / m)
        x2c = int((y2c - b) / m)
        return (x1c, y1c, x2c, y2c)

    return coords(lm, lb), coords(rm, rb), cross_y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Week 1 — [E] Temporal Smoother (unchanged)
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
#  Module-level state — one instance per process
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_smoother    = _Smoother(Cfg.SMOOTH_FRAMES)
_ema_cross   = _EMA(alpha=Cfg.EMA_ALPHA)
_calibrator  = _AutoCalibrator()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_overlay(frame, left, right, roi_polygon, calibrated):
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

    # ROI outline — yellow during warmup, green after calibration
    roi_color = (0, 200, 0) if calibrated else (0, 180, 255)
    cv2.polylines(out, roi_polygon, True, roi_color, 1)

    # Status label top-left
    status = "CAL" if calibrated else "WARM"
    cv2.putText(out, f"ROI:{status}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_lanes(frame):
    """
    Returns
    -------
    result   : BGR  — original frame with lane overlay
    hls_vis  : BGR  — HLS lane mask       (→ "edges" debug stream)
    roi_vis  : BGR  — ROI-masked edges    (→ "roi"   debug stream)
    """
    h, w = frame.shape[:2]

    # ── Week 1 preprocessing ───────────────────────────────────────
    masked_gray, hls_vis = hls_lane_mask(frame)       # [A]
    edges                = otsu_canny(masked_gray)     # [B]

    # ── Week 2 [F]+[H] — compute dynamic ROI top ──────────────────
    if Cfg.USE_AUTOCAL:
        top_y_frac = _calibrator.update(_ema_cross.value, h)
    else:
        # [F] EMA only — no auto-calibration
        ema_val = _ema_cross.value
        if ema_val is not None:
            safe_y     = ema_val + Cfg.CROSS_BUFFER_PX
            top_y_frac = float(np.clip(safe_y / h,
                                       Cfg.ROI_TOP_MIN,
                                       Cfg.ROI_TOP_MAX))
        else:
            top_y_frac = Cfg.ROI_TOP_Y   # fallback before first detection

    # ── [G] Pentagon or trapezoid ROI ─────────────────────────────
    roi_masked, roi_vis, roi_poly = apply_roi(edges, h, w, top_y_frac)

    # ── Hough ──────────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        roi_masked,
        rho           = Cfg.HOUGH_RHO,
        theta         = Cfg.HOUGH_THETA,
        threshold     = Cfg.HOUGH_THRESHOLD,
        minLineLength = Cfg.HOUGH_MIN_LENGTH,
        maxLineGap    = Cfg.HOUGH_MAX_GAP,
    )

    # ── [C]+[D] filter + cross clamp ──────────────────────────────
    left_raw, right_raw, cross_y = filter_and_fit(lines, h, w, top_y_frac)

    # ── [F] Update EMA with this frame's cross_y ──────────────────
    _ema_cross.update(cross_y)

    # ── [H] Feed calibrator ───────────────────────────────────────
    if Cfg.USE_AUTOCAL:
        _calibrator.update(cross_y, h)

    # ── [E] Temporal smooth ───────────────────────────────────────
    left, right = _smoother.update(left_raw, right_raw)

    result = draw_overlay(
        frame.copy(), left, right, roi_poly,
        calibrated=_calibrator.is_locked if Cfg.USE_AUTOCAL else True
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
        cv2.imshow("Lane Detection — Week 2", result)
        if debug:
            cv2.imshow("HLS Mask",  hls_vis)
            cv2.imshow("ROI Edges", roi_vis)
        else:
            for win in ("HLS Mask", "ROI Edges"):
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
    cv2.imshow("Lane Detection — Week 2", result)
    cv2.imshow("HLS Mask",                hls_vis)
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