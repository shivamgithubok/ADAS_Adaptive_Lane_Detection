"""
3D → 2D Cuboid Projection
==========================
Projects 3D cuboid corners back onto the image plane using the pinhole model.

Responsibilities:
    - Take 8 corners in 3D camera frame
    - Project each to 2D pixel coordinates
    - Return ProjectedCuboid with 2D corners and face connectivity
"""

import logging

import numpy as np

from adas.common.datatypes import (
    CameraIntrinsics,
    Cuboid3D,
    DepthEstimate,
    Detection2D,
    ProjectedCuboid,
)

logger = logging.getLogger("adas.perception.fusion")

# Face and edge connectivity for the cuboid
# Front face: corners [0, 1, 2, 3]
# Rear face:  corners [4, 5, 6, 7]
# Connecting edges: (0,4), (1,5), (2,6), (3,7)
FRONT_FACE_INDICES = [0, 1, 2, 3]
REAR_FACE_INDICES = [4, 5, 6, 7]
CONNECTING_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]


class CuboidProjector:
    """Projects 3D cuboid corners onto the 2D image plane."""

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        """
        Args:
            intrinsics: Camera intrinsic parameters (fx, fy, cx, cy).
        """
        self._intrinsics = intrinsics

        # Build 3×3 intrinsic matrix for batch projection
        self._K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        cuboid: Cuboid3D,
        detection: Detection2D,
        depth_estimate: DepthEstimate,
    ) -> ProjectedCuboid:
        """
        Project a 3D cuboid to 2D pixel coordinates.

        Args:
            cuboid: The 3D cuboid with 8 corners in camera frame.
            detection: The original 2D detection (for bbox reference).
            depth_estimate: The depth estimate (for metadata).

        Returns:
            ProjectedCuboid with 2D corner coordinates.
        """
        corners_2d = self._project_points(cuboid.corners_3d)

        return ProjectedCuboid(
            corners_2d=corners_2d,
            cuboid_3d=cuboid,
            bbox_2d=detection.bbox,
            depth_estimate=depth_estimate,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project an array of 3D points to 2D using the pinhole model.

        Args:
            points_3d: Shape (N, 3) — points in camera coordinates.

        Returns:
            Shape (N, 2) — pixel coordinates (u, v).
        """
        # points_3d: (N, 3) → transpose → (3, N)
        pts = points_3d.T  # (3, N)

        # Guard against zero or negative Z (behind camera)
        z = pts[2, :]
        z = np.where(z < 0.1, 0.1, z)  # Clamp to prevent division issues

        # Project: pixel = K @ (X/Z, Y/Z, 1)
        projected = self._K @ pts  # (3, N)
        projected[0, :] /= z
        projected[1, :] /= z

        # Return (N, 2) — (u, v) pixel coordinates
        return projected[:2, :].T.astype(np.float64)
