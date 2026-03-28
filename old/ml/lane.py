"""
lane_detection.py  —  Ultra-Fast Lane Detection v2 (UFLD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why UFLD instead of classical CV?
──────────────────────────────────
  Classical Hough + Sliding Window treats lanes as "collections of edge
  pixels". It has no concept of what a lane *is*, so cars, signs, shadows,
  and dashed lines all confuse it.

  UFLD treats lane detection as a row-based classification problem:
  "In each of the 18 horizontal rows of this image, which column
  contains the left / right lane?"  The model was trained on TuSimple
  and CULane — real highway dashcam footage with heavy traffic, dashed
  lines, partial occlusion, and varying lighting. It handles all of
  those cases because it learned from them.

Pipeline
─────────
  BGR frame
    │
    ▼
  Resize to 800x288 + ImageNet normalise   (preprocess_frame)
    │
    ▼
  UFLD backbone (ResNet-18) → row-anchor logits    (model forward)
    │
    ▼
  argmax per row → lane x-coordinates             (postprocess)
    │
    ▼
  Pick ego left + right lanes                     (_pick_ego_lanes)
    │
    ▼
  Temporal smoothing (LaneSmoother)
    │
    ▼
  Draw polylines + filled polygon on original frame (draw_lanes)

Setup — run once on your machine
──────────────────────────────────
  pip install torch torchvision
  pip install opencv-python numpy

  Model weights (~50 MB) are downloaded automatically on first run
  from the official UFLD GitHub release → ~/.cache/ufld/

Controls (standalone mode)
───────────────────────────
  q — quit
  d — toggle debug dots (one dot per row anchor)
  s — save current frame as PNG
"""

import os
import urllib.request
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# PyTorch — friendly error if missing
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as tvm
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Config:
    # Model input resolution (fixed by UFLD architecture)
    MODEL_W      = 800
    MODEL_H      = 288

    # 18 row anchors — evenly spaced in MODEL_H space (TuSimple training config)
    ROW_ANCHORS  = [121, 131, 141, 150, 160, 170, 180, 189,
                    199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

    # Column bins + lanes
    GRIDING_NUM  = 100
    NUM_LANES    = 4       # model outputs 4; we pick 2 ego lanes

    # Minimum row hits for a lane to be considered valid
    MIN_ROW_HITS = 5

    # Confidence threshold per row (softmax probability)
    CONF_THRESH  = 0.55

    # Weights
    WEIGHTS_DIR  = Path.home() / ".cache" / "ufld"
    WEIGHTS_URL  = (
        "https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2"
        "/releases/download/v2.0/tusimple_res18.pth"
    )
    WEIGHTS_FILE = Path.home() / ".cache" / "ufld" / "tusimple_res18.pth"

    # Drawing
    LANE_COLOR      = (0,  255,   0)
    FILL_COLOR      = (0,  200, 255)
    FILL_ALPHA      = 0.30
    LINE_THICKNESS  = 4
    POINT_RADIUS    = 5

    # Smoothing
    SMOOTH_FRAMES   = 8

    # ImageNet normalisation
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _UFLDHead(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(
            in_ch,
            Config.GRIDING_NUM * len(Config.ROW_ANCHORS) * Config.NUM_LANES
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x.view(-1, Config.NUM_LANES,
                      len(Config.ROW_ANCHORS), Config.GRIDING_NUM)


class UFLDNet(nn.Module):
    def __init__(self):
        super().__init__()
        base          = tvm.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.head     = _UFLDHead(in_ch=512)

    def forward(self, x):
        return self.head(self.backbone(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_model  = None
_device = None


def _download_weights():
    Config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if Config.WEIGHTS_FILE.exists():
        return
    print(f"[UFLD] Downloading weights to {Config.WEIGHTS_FILE} ...")
    urllib.request.urlretrieve(str(Config.WEIGHTS_URL), str(Config.WEIGHTS_FILE))
    print("[UFLD] Download complete.")


def _load_model():
    global _model, _device
    if _model is not None:
        return _model

    if not _TORCH_OK:
        raise RuntimeError(
            "\nPyTorch missing. Run:\n"
            "  pip install torch torchvision\nthen restart.\n"
        )

    _download_weights()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UFLD] Loading model on {_device} ...")

    net   = UFLDNet()
    state = torch.load(str(Config.WEIGHTS_FILE), map_location=_device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.to(_device).eval()

    _model = net
    print("[UFLD] Model ready.")
    return _model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pre / post processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_tfm = None


def _get_transform():
    global _tfm
    if _tfm is None:
        _tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
        ])
    return _tfm


def preprocess_frame(frame):
    """BGR ndarray → normalised float tensor (1, 3, H, W) on _device."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (Config.MODEL_W, Config.MODEL_H))
    return _get_transform()(resized).unsqueeze(0).to(_device)


def postprocess(logits, orig_h: int, orig_w: int):
    """
    logits shape : (1, NUM_LANES, NUM_ROWS, GRIDING_NUM)
    Returns      : list of NUM_LANES arrays, each (N, 2) int32 [x, y]
                   in original-frame pixel coordinates.
    """
    probs = torch.softmax(logits[0], dim=-1)   # (NUM_LANES, NUM_ROWS, G)
    lanes = []

    for li in range(Config.NUM_LANES):
        pts = []
        for ri, anchor_y in enumerate(Config.ROW_ANCHORS):
            col_bin    = int(torch.argmax(probs[li, ri]).item())
            confidence = float(probs[li, ri, col_bin].item())

            if confidence < Config.CONF_THRESH:
                continue

            x = int(col_bin / Config.GRIDING_NUM * orig_w)
            y = int(anchor_y / Config.MODEL_H    * orig_h)
            pts.append((x, y))

        lanes.append(
            np.array(pts, dtype=np.int32) if pts
            else np.empty((0, 2), dtype=np.int32)
        )

    return lanes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Temporal smoothing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LaneSmoother:
    """
    Average lane x-coordinates over the last N frames per row-anchor y.
    Removes single-frame jitter without introducing lag.
    """
    def __init__(self):
        self.history = deque(maxlen=Config.SMOOTH_FRAMES)

    def update(self, lanes):
        self.history.append(lanes)
        if len(self.history) == 1:
            return lanes

        smoothed = []
        for li in range(Config.NUM_LANES):
            y_to_xs = {}
            for past in self.history:
                if li >= len(past) or len(past[li]) < Config.MIN_ROW_HITS:
                    continue
                for x, y in past[li]:
                    y_to_xs.setdefault(y, []).append(x)

            if not y_to_xs:
                smoothed.append(np.empty((0, 2), dtype=np.int32))
                continue

            avg = sorted(
                [(int(np.mean(xs)), y) for y, xs in y_to_xs.items()],
                key=lambda p: p[1]
            )
            smoothed.append(np.array(avg, dtype=np.int32))

        return smoothed


_smoother = LaneSmoother()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lane selection + drawing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pick_ego_lanes(lanes, frame_w: int):
    """
    From up to 4 detected lanes pick the two ego lanes —
    the left and right lanes that straddle the image centre.
    Returns (left_pts, right_pts), either may be None.
    """
    valid = []
    for pts in lanes:
        if len(pts) < Config.MIN_ROW_HITS:
            continue
        # Representative x = median x of bottom third points
        bottom = pts[pts[:, 1] > pts[:, 1].max() * 0.7] if pts.ndim == 2 else pts
        rep_x  = int(np.median(bottom[:, 0])) if len(bottom) else int(np.median(pts[:, 0]))
        valid.append((rep_x, pts))

    if not valid:
        return None, None

    valid.sort(key=lambda c: c[0])
    cx = frame_w // 2

    left_cands  = [(x, p) for x, p in valid if x <= cx]
    right_cands = [(x, p) for x, p in valid if x >  cx]

    left_pts  = left_cands[-1][1]  if left_cands  else None
    right_pts = right_cands[0][1]  if right_cands else None

    # Fallback — if all lanes on one side, use closest two
    if left_pts is None and len(valid) >= 2:
        left_pts = valid[0][1]
    if right_pts is None and len(valid) >= 2:
        right_pts = valid[-1][1]
    elif right_pts is None and len(valid) == 1:
        right_pts = valid[0][1]

    return left_pts, right_pts


def draw_lanes(frame, lanes, show_debug: bool = False):
    """Draw ego-lane overlay on frame in-place."""
    orig_h, orig_w = frame.shape[:2]
    left_pts, right_pts = _pick_ego_lanes(lanes, orig_w)

    # Filled polygon
    if (left_pts  is not None and len(left_pts)  >= 2 and
        right_pts is not None and len(right_pts) >= 2):
        poly    = np.vstack([left_pts, right_pts[::-1]])
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], Config.FILL_COLOR)
        frame   = cv2.addWeighted(overlay, Config.FILL_ALPHA,
                                  frame,   1 - Config.FILL_ALPHA, 0)

    # Lane polylines
    for pts in (left_pts, right_pts):
        if pts is not None and len(pts) >= 2:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)],
                          False, Config.LANE_COLOR, Config.LINE_THICKNESS)

    # Debug: all lane dots
    if show_debug:
        colors = [(0,255,0),(0,200,255),(255,100,0),(255,0,200)]
        for li, pts in enumerate(lanes):
            for x, y in pts:
                cv2.circle(frame, (int(x), int(y)),
                           Config.POINT_RADIUS, colors[li % 4], -1)

    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API  (same signature as old classical version — app.py unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_lanes(frame, show_debug: bool = False):
    """
    UFLD lane detection.

    Returns 4-tuple (same as old version so app.py needs no change):
      result     — BGR frame with lane overlay
      debug_vis  — 800x288 model-input view
      edges      — blank placeholder (kept for stream compatibility)
      debug_win  — None
    """
    model = _load_model()
    h, w  = frame.shape[:2]

    tensor    = preprocess_frame(frame)
    with torch.no_grad():
        logits = model(tensor)

    raw_lanes    = postprocess(logits, h, w)
    smooth_lanes = _smoother.update(raw_lanes)
    result       = draw_lanes(frame.copy(), smooth_lanes, show_debug)

    debug_vis = cv2.resize(frame, (Config.MODEL_W, Config.MODEL_H))
    edges_placeholder = np.zeros((h, w), dtype=np.uint8)

    return result, debug_vis, edges_placeholder, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Standalone entry-points
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_video(source=0, show_debug=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return
    _load_model()
    print("Controls: q=quit | d=debug dots | s=save frame")
    debug = show_debug
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result, debug_vis, _, _ = detect_lanes(frame, debug)
        cv2.imshow("UFLD Lane Detection", result)
        if debug:
            cv2.imshow("Model input (800x288)", debug_vis)
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
    cv2.imshow("UFLD Lane Detection", result)
    cv2.imshow("Model input (800x288)", debug_vis)
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