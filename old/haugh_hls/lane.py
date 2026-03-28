"""
lane_detection.py — Week 1 Research Build
==========================================
Week 1 experiments:
  [A] HLS color mask     — white + yellow pixels only, grey cars filtered
  [B] Otsu adaptive Canny — per-frame lighting-aware thresholds
  [C] Cross-point clamp  — X cross geometrically impossible
  [D] RANSAC slope filter — outlier car edges auto-rejected
  [E] Temporal smoother  — 7-frame deque average, stable lines

detect_lanes() returns:
  result   — BGR overlay frame          ← app.py "result" stream
  hls_vis  — BGR HLS lane mask debug    ← app.py "edges" stream
  roi_vis  — BGR ROI edges debug        ← app.py "roi" stream

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
#  CONFIG — sirf yahan tune karo, baaki mat chheyna
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Cfg:

    # ── ROI ────────────────────────────────────────────────────────
    ROI_BOTTOM_LEFT_X  = 0.10
    ROI_BOTTOM_RIGHT_X = 0.90
    ROI_TOP_LEFT_X     = 0.43
    ROI_TOP_RIGHT_X    = 0.57
    ROI_TOP_Y          = 0.72   # raise (e.g. 0.78) if cross still visible

    # ── HLS color mask ─────────────────────────────────────────────
    WHITE_L_MIN   = 100    # lower = pick up dimmer lanes
    WHITE_L_MAX   = 250    # NEW — sky cut (sky L > 215, lane L < 215)
    WHITE_S_MAX   = 50     # NEW — sky has low S, but so does lane
                           #        use this ONLY if sky still bleeds in
    YELLOW_H_MIN  = 10
    YELLOW_H_MAX  = 35
    YELLOW_S_MIN  = 60

    # ── Blur ───────────────────────────────────────────────────────
    BLUR_KERNEL   = (5, 5)

    # ── Hough ──────────────────────────────────────────────────────
    HOUGH_RHO        = 1
    HOUGH_THETA      = np.pi / 180
    HOUGH_THRESHOLD  = 60
    HOUGH_MIN_LENGTH = 60
    HOUGH_MAX_GAP    = 200

    # ── Slope + x-position filter ──────────────────────────────────
    LEFT_SLOPE_MIN   = -0.9
    LEFT_SLOPE_MAX   = -0.4
    RIGHT_SLOPE_MIN  =  0.4
    RIGHT_SLOPE_MAX  =  0.9
    LEFT_X_MAX_FRAC  = 0.65
    RIGHT_X_MIN_FRAC = 0.35

    # ── Cross-point clamp ──────────────────────────────────────────
    CROSS_BUFFER_PX  = 30   # lines stop this many px below cross point

    # ── Temporal smoother ──────────────────────────────────────────
    SMOOTH_FRAMES    = 7

    # ── Drawing ────────────────────────────────────────────────────
    LANE_COLOR       = (0, 255,   0)
    FILL_COLOR       = (0, 200, 255)
    FILL_ALPHA       = 0.30
    LINE_THICKNESS   = 6
    ROI_BORDER_COLOR = (0, 255, 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [A] HLS Lane Mask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hls_lane_mask(frame):
    """
    Returns:
        masked_gray — grayscale with only white+yellow lane pixels
        hls_vis     — BGR visualisation of mask (for debug stream)
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H   = hls[:, :, 0].astype(np.int32)
    L   = hls[:, :, 1].astype(np.int32)
    S   = hls[:, :, 2].astype(np.int32)

    # White lane — L in range (not just > min, sky is too bright)
    white_mask  = ((L >= Cfg.WHITE_L_MIN) & (L <= Cfg.WHITE_L_MAX))

    # Yellow lane — specific hue + decent saturation
    yellow_mask = ((H >= Cfg.YELLOW_H_MIN) &
                   (H <= Cfg.YELLOW_H_MAX) &
                   (S >= Cfg.YELLOW_S_MIN))

    # ROI-based sky cut — zero out top 40% of frame entirely
    # Sky is always in top portion, lanes never are
    sky_cut = np.ones_like(L, dtype=np.uint8)
    sky_cut[:int(L.shape[0] * 0.40), :] = 0   # top 40% = black

    lane_mask = np.clip(
        white_mask.astype(np.uint8) + yellow_mask.astype(np.uint8),
        0, 1
    ).astype(np.uint8)

    # Apply sky cut — force top 40% to zero regardless of color
    lane_mask = (lane_mask * sky_cut * 255).astype(np.uint8)

    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, lane_mask)

    # BGR debug visualisation
    hls_vis = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    return masked_gray, hls_vis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [B] Otsu Adaptive Canny
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def otsu_canny(gray_image):
    """Per-frame adaptive thresholds using Otsu's method."""
    blurred = cv2.GaussianBlur(gray_image, Cfg.BLUR_KERNEL, 0)
    otsu_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_low  = max(10, int(otsu_val * 0.5))
    t_high = max(20, int(otsu_val))
    return cv2.Canny(blurred, t_low, t_high)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_roi(edges, h, w):
    mask  = np.zeros_like(edges)
    top_y = int(h * Cfg.ROI_TOP_Y)
    polygon = np.array([[
        (int(w * Cfg.ROI_BOTTOM_LEFT_X),  h    ),
        (int(w * Cfg.ROI_TOP_LEFT_X),     top_y),
        (int(w * Cfg.ROI_TOP_RIGHT_X),    top_y),
        (int(w * Cfg.ROI_BOTTOM_RIGHT_X), h    ),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    # BGR debug visualisation
    roi_vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

    return masked, roi_vis, polygon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [D] RANSAC / mean slope filter  +  [C] Cross-point clamp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fit(lines_group):
    if len(lines_group) < 2 or not _RANSAC_AVAILABLE:
        return tuple(np.mean(lines_group, axis=0))
    slopes     = np.array([s for s, _ in lines_group]).reshape(-1, 1)
    intercepts = np.array([b for _, b in lines_group])
    try:
        reg = RANSACRegressor(min_samples=2, residual_threshold=10)
        reg.fit(slopes, intercepts)
        inliers = [lines_group[i]
                   for i, ok in enumerate(reg.inlier_mask_) if ok]
        return tuple(np.mean(inliers, axis=0)) if inliers else \
               tuple(np.mean(lines_group, axis=0))
    except Exception:
        return tuple(np.mean(lines_group, axis=0))


def filter_and_fit(lines, h, w):
    ls, li = [], []
    rs, ri = [], []

    if lines is None:
        return None, None

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

    # [C] cross-point clamp
    cross_y = None
    if lm and rm and abs(lm - rm) > 1e-6:
        xc      = (rb - lb) / (lm - rm)
        cross_y = int(lm * xc + lb)

    def coords(m, b):
        if m is None:
            return None
        y1c = h
        y2c = int(h * Cfg.ROI_TOP_Y)
        if cross_y is not None:
            y2c = max(y2c, cross_y + Cfg.CROSS_BUFFER_PX)
        x1c = int((y1c - b) / m)
        x2c = int((y2c - b) / m)
        return (x1c, y1c, x2c, y2c)

    return coords(lm, lb), coords(rm, rb)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [E] Temporal Smoother
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


_smoother = _Smoother(Cfg.SMOOTH_FRAMES)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_overlay(frame, left, right, roi_polygon):
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

    cv2.polylines(out, roi_polygon, True, Cfg.ROI_BORDER_COLOR, 1)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pipeline — called by app.py every frame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_lanes(frame):
    """
    Returns
    -------
    result   : BGR  — original frame with lane overlay
    hls_vis  : BGR  — white/yellow lane mask  (→ "edges" debug stream)
    roi_vis  : BGR  — ROI-masked Canny edges  (→ "roi"   debug stream)
    """
    h, w = frame.shape[:2]

    masked_gray, hls_vis          = hls_lane_mask(frame)         # A
    edges                         = otsu_canny(masked_gray)       # B
    roi_masked, roi_vis, roi_poly = apply_roi(edges, h, w)

    lines = cv2.HoughLinesP(
        roi_masked,
        rho           = Cfg.HOUGH_RHO,
        theta         = Cfg.HOUGH_THETA,
        threshold     = Cfg.HOUGH_THRESHOLD,
        minLineLength = Cfg.HOUGH_MIN_LENGTH,
        maxLineGap    = Cfg.HOUGH_MAX_GAP,
    )

    left_raw, right_raw = filter_and_fit(lines, h, w)             # C + D
    left, right         = _smoother.update(left_raw, right_raw)   # E

    result = draw_overlay(frame.copy(), left, right, roi_poly)

    # ALL three returns are BGR — app.py must NOT call cvtColor on them
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
        cv2.imshow("Lane Detection", result)
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
    cv2.imshow("Lane Detection", result)
    cv2.imshow("HLS Mask",       hls_vis)
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