"""
3D Cuboid Generator
===================
Generates a 3D bounding cuboid from a 2D detection and its depth estimate.

Responsibilities:
    - Backproject the 2D bounding box center to a 3D point using pinhole geometry
    - Look up class dimension priors (length, width, height)
    - Generate 8 cuboid corner vertices in the camera coordinate frame
    - Return a Cuboid3D dataclass

Coordinate convention (camera frame):
    X → right
    Y → down
    Z → forward (into the scene)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from adas.common.datatypes import (
    CameraIntrinsics,
    Cuboid3D,
    DepthEstimate,
    Detection2D,
)

logger = logging.getLogger("adas.perception.fusion")


class CuboidGenerator:
    """Generates 3D cuboids from 2D detections + depth estimates."""

    def __init__(
        self,
        class_priors: Dict[str, Dict[str, float]],
        intrinsics: CameraIntrinsics,
    ) -> None:
        """
        Args:
            class_priors: Per-class dimension priors from config, e.g.
                          {"car": {"length": 4.5, "width": 1.8, "height": 1.5}, ...}
            intrinsics: Camera intrinsic parameters (fx, fy, cx, cy).
        """
        self._priors = class_priors
        self._intrinsics = intrinsics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        detection: Detection2D,
        depth_estimate: DepthEstimate,
    ) -> Optional[Cuboid3D]:
        """
        Generate a 3D cuboid for a single detected object.

        Args:
            detection: The 2D detection (class, bbox, confidence).
            depth_estimate: Depth statistics for this object's ROI.

        Returns:
            Cuboid3D dataclass, or None if the class has no priors.
        """
        # Look up class priors
        dims = self._get_dimensions(detection.class_name)
        if dims is None:
            logger.debug(
                "No dimension priors for class '%s', skipping cuboid generation",
                detection.class_name,
            )
            return None

        length, width, height = dims
        # Calculate bounding box width in pixels
        x1, y1, x2, y2 = detection.bbox
        bbox_width_2d = max(1.0, float(x2 - x1))
        
        # Calculate metric Z distance using pinhole geometry:
        # Z = (real_width_meters * focal_length) / pixel_width
        Z = (width * self._intrinsics.fx) / bbox_width_2d

        # Limit to 25.0 meters to focus on nearby cars (and prevent far noise)
        if Z > 25.0:
            logger.debug("Object too far (%.2f m), skipping", Z)
            return None
            
        if Z <= 0.5:
            logger.debug("Depth too small (%.2f m), skipping", Z)
            return None

        # Backproject 2D bbox center → 3D
        cx_2d = (x1 + x2) / 2.0
        cy_2d = (y1 + y2) / 2.0

        X = (cx_2d - self._intrinsics.cx) * Z / self._intrinsics.fx

        # Adjust Y so the cuboid bottom aligns with the detected bottom edge.
        # The bottom of the bbox roughly corresponds to the ground contact,
        # so we shift the center upward by half the height.
        bottom_y_3d = (y2 - self._intrinsics.cy) * Z / self._intrinsics.fy
        Y = bottom_y_3d - height / 2.0

        center_3d = np.array([X, Y, Z], dtype=np.float64)

        # Generate 8 corners
        corners_3d = self._make_cuboid_corners(center_3d, length, width, height)

        return Cuboid3D(
            center_3d=center_3d,
            dimensions=(length, width, height),
            corners_3d=corners_3d,
            class_name=detection.class_name,
            confidence=detection.confidence,
            distance=Z,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_dimensions(
        self, class_name: str
    ) -> Optional[Tuple[float, float, float]]:
        """Retrieve (length, width, height) for a class name."""
        key = class_name.lower()
        if key not in self._priors:
            return None
        p = self._priors[key]
        return (p["length"], p["width"], p["height"])

    @staticmethod
    def _make_cuboid_corners(
        center: np.ndarray,
        length: float,
        width: float,
        height: float,
    ) -> np.ndarray:
        """
        Generate 8 cuboid corners centered at `center`.

        Corner ordering (camera frame: X-right, Y-down, Z-forward):
            Front face (closer Z):  0=TL, 1=TR, 2=BR, 3=BL
            Rear face  (farther Z): 4=TL, 5=TR, 6=BR, 7=BL

        Returns:
            np.ndarray of shape (8, 3).
        """
        hl = length / 2.0  # half-length along Z (forward/back)
        hw = width / 2.0   # half-width along X (left/right)
        hh = height / 2.0  # half-height along Y (up/down)

        # fmt: off
        offsets = np.array([
            [-hw, -hh, -hl],  # 0: front-top-left
            [ hw, -hh, -hl],  # 1: front-top-right
            [ hw,  hh, -hl],  # 2: front-bottom-right
            [-hw,  hh, -hl],  # 3: front-bottom-left
            [-hw, -hh,  hl],  # 4: rear-top-left
            [ hw, -hh,  hl],  # 5: rear-top-right
            [ hw,  hh,  hl],  # 6: rear-bottom-right
            [-hw,  hh,  hl],  # 7: rear-bottom-left
        ], dtype=np.float64)
        # fmt: on

        return center + offsets
