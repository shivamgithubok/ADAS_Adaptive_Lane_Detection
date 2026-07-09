#!/usr/bin/env python3
"""Quick diagnostic: print raw depth map stats for a single frame."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from adas.perception.depth.depth_engine import DepthEngine

video_path = PROJECT_ROOT / "adas" / "test-video_480.mp4"
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame"); sys.exit(1)

engine = DepthEngine(
    model_path=PROJECT_ROOT / "depth_anything_v2_vits.pth",
    encoder="vits",
    repo_path=PROJECT_ROOT / "depth_anything_v2_repo",
)

depth = engine.infer(frame)
print(f"Frame shape: {frame.shape}")
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
print(f"  min={depth.min():.4f}  max={depth.max():.4f}  mean={depth.mean():.4f}  median={np.median(depth):.4f}")
print(f"  5th pct={np.percentile(depth,5):.4f}  95th pct={np.percentile(depth,95):.4f}")

# Check a far region (top half = sky/horizon) vs near region (bottom quarter = road)
h = depth.shape[0]
far_region = depth[:h//3, :]
near_region = depth[2*h//3:, :]
print(f"\nFar region (top third):  mean={far_region.mean():.4f}  median={np.median(far_region):.4f}")
print(f"Near region (bot third): mean={near_region.mean():.4f}  median={np.median(near_region):.4f}")

if near_region.mean() > far_region.mean():
    print("\n>> HIGHER values = CLOSER (disparity-like) — NEED INVERSION")
else:
    print("\n>> HIGHER values = FARTHER (depth-like) — no inversion needed")
