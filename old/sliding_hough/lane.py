import cv2
import numpy as np
from collections import deque


# ─── Config — sab tunable values ek jagah ────────────────────────────────────

class Config:
    # ROI — top edge neeche rakho taaki car andar na aaye
    ROI_TOP_PCT        = 0.72   # 0.68 → 0.72  (cross band karo + more road visible)
    ROI_BOTTOM_LEFT    = 0.05   # 0.10 → 0.05  (wider bottom = more lane pixels)
    ROI_BOTTOM_RIGHT   = 0.95   # 0.90 → 0.95
    ROI_TOP_LEFT       = 0.42
    ROI_TOP_RIGHT      = 0.58

    # Sliding window
    N_WINDOWS          = 12     # 9 → 12  (finer strips, dashed lines better cover)
    WINDOW_MARGIN      = 100    # 80 → 100  (wider search per window)
    MIN_PIXELS         = 150    # 500 → 150  ← ASLI PROBLEM YEH THA, bahut strict

    # Temporal smoothing — last N frames ka average
    SMOOTH_FRAMES      = 10

    # Preprocessing
    BLUR_KERNEL        = (5, 5)
    CANNY_LOW          = 30     # 50 → 30  (faint markings bhi catch hongi)
    CANNY_HIGH         = 120    # 150 → 120

    # Drawing
    LANE_COLOR         = (0, 255, 0)    # green lines
    FILL_COLOR         = (0, 200, 255)  # cyan fill
    FILL_ALPHA         = 0.3
    LINE_THICKNESS     = 8


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, Config.BLUR_KERNEL, 0)
    edges   = cv2.Canny(blurred, Config.CANNY_LOW, Config.CANNY_HIGH)
    return edges


def get_roi(edges):
    h, w  = edges.shape
    top_y = int(h * Config.ROI_TOP_PCT)

    polygon = np.array([[
        (int(w * Config.ROI_BOTTOM_LEFT),  h),
        (int(w * Config.ROI_TOP_LEFT),     top_y),
        (int(w * Config.ROI_TOP_RIGHT),    top_y),
        (int(w * Config.ROI_BOTTOM_RIGHT), h),
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask), polygon


# ─── Sliding Window ───────────────────────────────────────────────────────────
#
#  Hough ki jagah yeh approach use karte hain:
#  1. Frame ko N horizontal strips mein divide karo
#  2. Har strip mein white pixels ka centroid nikalo
#  3. Centroids se lane position track karo
#  Car ki edges lane pixels ke saath mix nahi hoti
#  kyunki woh lane ke upar hoti hain, lane ke andar nahi

def sliding_window(roi_binary, debug=False):
    h, w = roi_binary.shape

    # ── Starting position: bottom half ka histogram ──
    # Histogram = har column mein kitne white pixels hain
    # Lane markings ke neeche 2 peaks honge — left aur right
    histogram   = np.sum(roi_binary[h // 2:, :], axis=0)
    midpoint    = w // 2

    # Fallback: agar histogram mein koi strong peak nahi toh
    # frame ke 25% aur 75% pe default starting point set karo
    left_peak  = int(np.argmax(histogram[:midpoint]))
    right_peak = int(np.argmax(histogram[midpoint:]) + midpoint)

    left_base_x  = left_peak  if histogram[left_peak]  > 50 else int(w * 0.25)
    right_base_x = right_peak if histogram[right_peak] > 50 else int(w * 0.75)

    window_h  = h // Config.N_WINDOWS
    cur_left  = left_base_x
    cur_right = right_base_x

    left_pts  = []   # (x, y) centroids for left lane
    right_pts = []   # (x, y) centroids for right lane

    debug_frame = None
    if debug:
        debug_frame = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2BGR)

    for i in range(Config.N_WINDOWS):
        # Bottom se upar jaate hain
        y_low  = h - (i + 1) * window_h
        y_high = h - i * window_h
        y_mid  = (y_low + y_high) // 2

        # Left window boundaries
        lx_low  = max(0, cur_left  - Config.WINDOW_MARGIN)
        lx_high = min(w, cur_left  + Config.WINDOW_MARGIN)
        # Right window boundaries
        rx_low  = max(0, cur_right - Config.WINDOW_MARGIN)
        rx_high = min(w, cur_right + Config.WINDOW_MARGIN)

        # White pixels nikalo
        left_win  = roi_binary[y_low:y_high, lx_low:lx_high]
        right_win = roi_binary[y_low:y_high, rx_low:rx_high]

        left_count  = np.sum(left_win  > 0)
        right_count = np.sum(right_win > 0)

        # Enough pixels hain toh centroid nikalo aur center update karo
        if left_count > Config.MIN_PIXELS:
            col_sum  = np.sum(left_win, axis=0)
            centroid = lx_low + int(np.argmax(col_sum))
            cur_left = centroid
            left_pts.append((centroid, y_mid))

        if right_count > Config.MIN_PIXELS:
            col_sum  = np.sum(right_win, axis=0)
            centroid = rx_low + int(np.argmax(col_sum))
            cur_right = centroid
            right_pts.append((centroid, y_mid))

        # Debug: windows draw karo
        if debug and debug_frame is not None:
            cv2.rectangle(debug_frame,
                          (lx_low, y_low), (lx_high, y_high),
                          (255, 100, 0), 1)
            cv2.rectangle(debug_frame,
                          (rx_low, y_low), (rx_high, y_high),
                          (0, 100, 255), 1)

    return left_pts, right_pts, debug_frame


# ─── Centroids se Line fit karo ───────────────────────────────────────────────

def fit_lane_line(frame, points):
    """
    Centroids pe linear regression lagao → ek clean lane line milegi
    Numpy polyfit use karta hai — simple aur fast
    """
    if len(points) < 2:
        return None

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)

    try:
        # degree=1 → straight line fit
        coeffs = np.polyfit(ys, xs, 1)   # x = m*y + b (y ke against fit karo)
    except np.linalg.LinAlgError:
        return None

    h = frame.shape[0]
    y1 = h
    y2 = int(h * Config.ROI_TOP_PCT)

    x1 = int(np.polyval(coeffs, y1))
    x2 = int(np.polyval(coeffs, y2))

    # Sanity check — line frame ke bahar nahi jaani chahiye
    frame_w = frame.shape[1]
    if not (0 <= x1 <= frame_w and 0 <= x2 <= frame_w):
        return None

    return (x1, y1, x2, y2)


# ─── Temporal Smoothing ───────────────────────────────────────────────────────

left_history  = deque(maxlen=Config.SMOOTH_FRAMES)
right_history = deque(maxlen=Config.SMOOTH_FRAMES)


def smooth_line(history, new_line):
    """
    Last N frames ka average — jitter aur hallucination band hoti hai
    Ek frame mein car edge detect ho bhi jaaye toh average shift nahi hoga
    """
    if new_line is not None:
        history.append(new_line)
    if not history:
        return None
    avg = np.mean(history, axis=0)
    return tuple(map(int, avg))


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_lanes(frame, left_line, right_line):
    line_layer = np.zeros_like(frame)
    overlay    = frame.copy()

    if left_line:
        cv2.line(line_layer,
                 (left_line[0],  left_line[1]),
                 (left_line[2],  left_line[3]),
                 Config.LANE_COLOR, Config.LINE_THICKNESS)

    if right_line:
        cv2.line(line_layer,
                 (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 Config.LANE_COLOR, Config.LINE_THICKNESS)

    # Lane ke beech fill
    if left_line and right_line:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], Config.FILL_COLOR)
        frame = cv2.addWeighted(overlay, Config.FILL_ALPHA,
                                frame,   1 - Config.FILL_ALPHA, 0)

    return cv2.addWeighted(frame, 1.0, line_layer, 1.0, 0)


def draw_roi_outline(frame, polygon):
    """ROI boundary dikhao — debug ke liye useful"""
    cv2.polylines(frame, polygon, True, (0, 255, 255), 1)
    return frame


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def detect_lanes(frame, show_debug=False):
    edges           = preprocess(frame)
    roi, polygon    = get_roi(edges)

    left_pts, right_pts, debug_win = sliding_window(roi, debug=show_debug)

    left_raw   = fit_lane_line(frame, left_pts)
    right_raw  = fit_lane_line(frame, right_pts)

    left_final  = smooth_line(left_history,  left_raw)
    right_final = smooth_line(right_history, right_raw)

    result = draw_lanes(frame, left_final, right_final)

    return result, roi, edges, debug_win


# ─── Entry Point ─────────────────────────────────────────────────────────────

def process_video(source=0, show_debug=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    print("Lane Detection (Sliding Window) — 'q' quit | 'd' debug | 'r' ROI outline")
    debug   = show_debug
    show_roi = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, roi, edges, debug_win = detect_lanes(frame, debug)

        if show_roi:
            _, polygon = get_roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            result = draw_roi_outline(result, polygon)

        cv2.imshow("Lane Detection — Sliding Window", result)

        if debug:
            cv2.imshow("Canny Edges", edges)
            cv2.imshow("ROI (masked)", roi)
            if debug_win is not None:
                cv2.imshow("Sliding Windows", debug_win)
        else:
            for w in ("Canny Edges", "ROI (masked)", "Sliding Windows"):
                try: cv2.destroyWindow(w)
                except: pass

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('d'): debug    = not debug
        elif key == ord('r'): show_roi = not show_roi

    cap.release()
    cv2.destroyAllWindows()


def process_image(path, show_debug=False):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: {path}")
        return

    result, roi, edges, debug_win = detect_lanes(frame, show_debug)

    cv2.imshow("Lane Detection", result)
    cv2.imshow("Canny Edges",   edges)
    cv2.imshow("ROI Masked",    roi)
    if debug_win is not None:
        cv2.imshow("Sliding Windows", debug_win)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Quick Tuning Guide ───────────────────────────────────────────────────────
#
#  Problem                         Fix
#  ─────────────────────────────── ──────────────────────────────────────────
#  Cross still visible             Config.ROI_TOP_PCT badhao (0.68 → 0.72)
#  Lines too short / not reaching  Config.ROI_TOP_PCT ghatao (0.68 → 0.64)
#  Jittery / unstable lines        Config.SMOOTH_FRAMES badhao (10 → 15)
#  Missing dashed lane lines       Config.MIN_PIXELS ghatao (500 → 300)
#  Car edges still detected        Config.ROI_TOP_PCT badhao + MIN_PIXELS badhao
#  Windows too narrow              Config.WINDOW_MARGIN badhao (80 → 100)
#  Too many false detections       Config.N_WINDOWS badhao (9 → 12)

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    debug = "--debug" in args
    src_args = [a for a in args if not a.startswith("--")]

    if src_args:
        src = src_args[0]
        if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            process_image(src, debug)
        else:
            process_video(src, debug)
    else:
        process_video(0, debug)