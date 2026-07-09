"""
Object Depth Estimator
======================
Extracts depth statistics from a dense depth map for each detected object.

Responsibilities:
    - Crop depth ROI from the full depth map using the YOLO bounding box
    - Shrink ROI to avoid edge contamination
    - Compute median, mean, variance over valid depth pixels
    - Return DepthEstimate in meters (depth map is already metric)
"""

import logging
from typing import Optional

import numpy as np

from adas.common.datatypes import DepthEstimate

logger = logging.getLogger("adas.perception.fusion")


class ObjectDepthEstimator:
    """Extracts per-object depth statistics from a dense depth map."""

    def __init__(
        self,
        roi_shrink: float = 0.15,
        min_valid_pixels: int = 10,
    ) -> None:
        """
        Args:
            roi_shrink: Fraction to shrink the bounding box ROI on each side
                        (0.15 = 15% inset to avoid background contamination).
            min_valid_pixels: Minimum number of non-zero depth pixels required
                             to produce a valid estimate.
        """
        self._roi_shrink = roi_shrink
        self._min_valid_pixels = min_valid_pixels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        depth_map: np.ndarray,
        bbox: tuple,
    ) -> Optional[DepthEstimate]:
        """
        Estimate depth statistics for a single detected object.

        Args:
            depth_map: Full-frame depth map (H, W), float32, in meters.
            bbox: YOLO bounding box as (x1, y1, x2, y2) in pixels.

        Returns:
            DepthEstimate dataclass, or None if insufficient valid pixels.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.debug("Invalid bbox after clamping: (%d,%d,%d,%d)", x1, y1, x2, y2)
            return None

        # Shrink ROI to avoid edge contamination
        bw = x2 - x1
        bh = y2 - y1
        margin_x = int(bw * self._roi_shrink)
        margin_y = int(bh * self._roi_shrink)

        rx1 = x1 + margin_x
        ry1 = y1 + margin_y
        rx2 = x2 - margin_x
        ry2 = y2 - margin_y

        if rx2 <= rx1 or ry2 <= ry1:
            # Fallback to original bbox if too small after shrinking
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        # Extract depth ROI
        roi = depth_map[ry1:ry2, rx1:rx2]
        valid_mask = roi > 0.5  # Filter out very close / invalid depths
        valid_depths = roi[valid_mask]

        if len(valid_depths) < self._min_valid_pixels:
            logger.debug(
                "Insufficient valid pixels (%d) for bbox (%d,%d,%d,%d)",
                len(valid_depths), x1, y1, x2, y2,
            )
            return None

        # Compute statistics — depth map is already in meters
        median_depth = float(np.median(valid_depths))
        mean_depth = float(np.mean(valid_depths))
        depth_variance = float(np.var(valid_depths))

        return DepthEstimate(
            median_depth=median_depth,
            mean_depth=mean_depth,
            depth_variance=depth_variance,
            valid_pixels=int(len(valid_depths)),
        )
