import cv2
import numpy as np

def format_bbox(bbox):
    """Format bounding box (x1, y1, x2, y2) to integers."""
    return tuple(map(int, bbox))

def draw_transparent_mask(frame, mask_pts, color=(0, 255, 0), alpha=0.4):
    """Draw a transparent polygon mask on the frame."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(mask_pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
