import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import RANSACRegressor


left_history  = deque(maxlen=7)
right_history = deque(maxlen=7)

# ─── Core helpers ────────────────────────────────────────────────────────────

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_blur(gray, kernel_size=(5, 5)):
    return cv2.GaussianBlur(gray, kernel_size, 0)


def detect_edges(blurred, low_threshold=50, high_threshold=150):
    return cv2.Canny(blurred, low_threshold, high_threshold)


def region_of_interest(edges, height, width):
    """
    Mask out everything except a trapezoidal region at the bottom of the frame
    where lane lines are typically found.
    """
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1),  height),
        (int(width * 0.41), int(height * 0.62)),
        (int(width * 0.55), int(height * 0.62)),
        (int(width * 0.9),  height),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def hough_lines(roi):
    """Run probabilistic Hough Transform to find line segments."""
    return cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=90,
        maxLineGap=200,
    )


# ─── Line averaging ───────────────────────────────────────────────────────────

def _fit_ransac(lines_group):
    """
    np.mean ki jagah RANSAC — outlier car edges / shadows auto-reject hoti hain.
    Returns (slope, intercept).
    """
    if len(lines_group) < 2:
        return lines_group[0]

    slopes     = np.array([s for s, _ in lines_group]).reshape(-1, 1)
    intercepts = np.array([i for _, i in lines_group])

    try:
        ransac = RANSACRegressor(min_samples=2, residual_threshold=10)
        ransac.fit(slopes, intercepts)
        inliers = [lines_group[i] for i, ok in enumerate(ransac.inlier_mask_) if ok]
        return tuple(np.mean(inliers, axis=0))
    except Exception:
        return tuple(np.mean(lines_group, axis=0))   # fallback


def average_slope_intercept(frame, lines):
    """
    Separate detected line segments into left / right lanes by slope sign,
    then fit a single representative line using RANSAC (outlier-robust).
    """
    height, width = frame.shape[:2]          # ← fix: width defined here
    left_lines, right_lines = [], []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # tighter slope range — car edges bahut flat/steep hoti hain
        if -0.9 < slope < -0.4:              # left lane
            if x1 < width * 0.65:            # x-sanity check
                left_lines.append((slope, intercept))
        elif 0.4 < slope < 0.9:             # right lane
            if x1 > width * 0.35:            # x-sanity check
                right_lines.append((slope, intercept))

    def make_line(lines_group):
        if not lines_group:
            return None
        slope, intercept = _fit_ransac(lines_group)
        y1 = height
        y2 = int(height * 0.62)              # hard clamp — X cross band
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    left  = make_line(left_lines)
    right = make_line(right_lines)
    return left, right


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_lane_overlay(frame, left_line, right_line, alpha=0.3):
    """
    Draw solid lane lines and a translucent filled polygon between them.
    """
    overlay = frame.copy()
    line_image = np.zeros_like(frame)

    if left_line is not None:
        cv2.line(line_image, (left_line[0],  left_line[1]),
                              (left_line[2],  left_line[3]),  (0, 255, 0), 8)
    if right_line is not None:
        cv2.line(line_image, (right_line[0], right_line[1]),
                              (right_line[2], right_line[3]), (0, 255, 0), 8)

    # Filled lane polygon
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return cv2.addWeighted(frame, 1.0, line_image, 1.0, 0)


# ─── Full pipeline (single frame) ────────────────────────────────────────────

def detect_lanes(frame):
    height, width = frame.shape[:2]

    gray    = to_grayscale(frame)
    blurred = apply_blur(gray)
    edges   = detect_edges(blurred)
    roi     = region_of_interest(edges, height, width)
    lines   = hough_lines(roi)
    left, right = average_slope_intercept(frame, lines)

    def smooth_line(history, new_line):
        if new_line is not None:
            history.append(new_line)
        if not history:
            return None
        return tuple(map(int, np.mean(history, axis=0)))

    left  = smooth_line(left_history,  left)
    right = smooth_line(right_history, right)


    result  = draw_lane_overlay(frame, left, right)
    return result, roi, edges            # also return debug frames


# ─── Entry point ─────────────────────────────────────────────────────────────

def process_video(source=0, show_debug=False):
    """
    Run lane detection on a video file or webcam feed.

    Args:
        source: path to a video file, or 0 for the default webcam.
        show_debug: show intermediate edge / ROI windows when True.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    print("Lane detection running — press 'q' to quit, 'd' to toggle debug view.")
    debug = show_debug

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, roi, edges = detect_lanes(frame)

        cv2.imshow("Lane Detection", result)

        if debug:
            cv2.imshow("Edges",        edges)
            cv2.imshow("ROI (masked)", roi)
        else:
            for win in ("Edges", "ROI (masked)"):
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


def process_image(path):
    """Run lane detection on a single image file and display the result."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    result, roi, edges = detect_lanes(frame)

    cv2.imshow("Original",       frame)
    cv2.imshow("Edges",          edges)
    cv2.imshow("ROI (masked)",   roi)
    cv2.imshow("Lane Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        src = sys.argv[1]
        # Detect whether the argument is an image extension
        if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            process_image(src)
        else:
            # treat as video file
            process_video(source=src, show_debug="--debug" in sys.argv)
    else:
        # Default: webcam
        process_video(source=0, show_debug="--debug" in sys.argv)