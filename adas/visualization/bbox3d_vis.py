"""
3D Bounding Box Visualizer
===========================
Draws projected 3D cuboids, distance labels, and optional depth overlays
onto RGB frames in an ADAS-style visual presentation.

Responsibilities:
    - Draw front face (filled polygon, semi-transparent)
    - Draw rear face (thinner outline)
    - Draw connecting edges between front and rear
    - Draw distance label and class name
    - Draw the original 2D YOLO bounding box (thin line)
    - Optional depth map colormap overlay
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from adas.common.datatypes import ProjectedCuboid
from adas.perception.fusion.projection import (
    CONNECTING_EDGES,
    FRONT_FACE_INDICES,
    REAR_FACE_INDICES,
)

logger = logging.getLogger("adas.visualization.bbox3d")

# Default ADAS color palette (BGR)
DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 200, 0),
    "truck": (0, 140, 255),
    "bus": (0, 255, 255),
    "person": (0, 255, 100),
    "motorcycle": (255, 0, 200),
    "bicycle": (100, 255, 0),
    "default": (200, 200, 200),
}


class BBox3DVisualizer:
    """Renders projected 3D cuboids onto BGR frames."""

    def __init__(
        self,
        color_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
        front_alpha: float = 0.25,
        front_thickness: int = 2,
        rear_thickness: int = 1,
        edge_thickness: int = 1,
        label_font_scale: float = 0.55,
        label_thickness: int = 2,
        label_bg_alpha: float = 0.6,
    ) -> None:
        """
        Args:
            color_palette: Per-class BGR color mapping.
            front_alpha: Opacity for the front face fill.
            front_thickness: Line thickness for the front face outline.
            rear_thickness: Line thickness for the rear face outline.
            edge_thickness: Line thickness for connecting edges.
            label_font_scale: Font scale for the distance/class label.
            label_thickness: Font thickness for labels.
            label_bg_alpha: Opacity for the label background.
        """
        self._colors = color_palette or DEFAULT_COLORS
        self._front_alpha = front_alpha
        self._front_thickness = front_thickness
        self._rear_thickness = rear_thickness
        self._edge_thickness = edge_thickness
        self._label_font_scale = label_font_scale
        self._label_thickness = label_thickness
        self._label_bg_alpha = label_bg_alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_all(
        self,
        frame: np.ndarray,
        projected_cuboids: List[ProjectedCuboid],
        depth_map: Optional[np.ndarray] = None,
        depth_colormap: int = cv2.COLORMAP_JET,
        depth_alpha: float = 0.15,
        show_depth_overlay: bool = False,
    ) -> np.ndarray:
        """
        Draw all projected cuboids onto a frame.

        Args:
            frame: BGR image (H, W, 3), modified in place.
            projected_cuboids: List of ProjectedCuboid objects.
            depth_map: Optional dense depth map for background overlay.
            depth_colormap: OpenCV colormap ID for depth visualization.
            depth_alpha: Alpha blend factor for depth overlay.
            show_depth_overlay: Whether to show depth map as background.

        Returns:
            The annotated frame.
        """
        output = frame.copy()

        # Optional depth overlay
        if show_depth_overlay and depth_map is not None:
            output = self._draw_depth_overlay(
                output, depth_map, depth_colormap, depth_alpha
            )

        # Draw each cuboid
        for pc in projected_cuboids:
            output = self.draw(output, pc)

        return output

    def draw(self, frame: np.ndarray, pc: ProjectedCuboid) -> np.ndarray:
        """
        Draw a single projected cuboid onto a frame.

        Args:
            frame: BGR image, modified in place.
            pc: ProjectedCuboid to draw.

        Returns:
            The annotated frame.
        """
        corners = pc.corners_2d.astype(np.int32)
        color = self._get_color(pc.cuboid_3d.class_name)

        # 1. Draw connecting edges (back-to-front depth ordering)
        self._draw_connecting_edges(frame, corners, color)

        # 2. Draw rear face
        self._draw_face(
            frame,
            corners,
            REAR_FACE_INDICES,
            color,
            self._rear_thickness,
            fill=False,
        )

        # 3. Draw front face (with semi-transparent fill)
        self._draw_face(
            frame,
            corners,
            FRONT_FACE_INDICES,
            color,
            self._front_thickness,
            fill=True,
        )

        # 4. Draw YOLO 2D bounding box (thin dotted)
        self._draw_bbox_2d(frame, pc.bbox_2d, color)

        # 5. Draw label
        self._draw_label(frame, pc, color)

        return frame

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_face(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        indices: List[int],
        color: Tuple[int, int, int],
        thickness: int,
        fill: bool = False,
    ) -> None:
        """Draw a cuboid face (4 corners) as a polygon."""
        pts = corners[indices].reshape((-1, 1, 2))

        if fill:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [corners[indices]], color)
            cv2.addWeighted(overlay, self._front_alpha, frame, 1 - self._front_alpha, 0, frame)

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def _draw_connecting_edges(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw 4 edges connecting front face to rear face."""
        for i, j in CONNECTING_EDGES:
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[j])
            cv2.line(frame, pt1, pt2, color, self._edge_thickness)

    def _draw_bbox_2d(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw the original YOLO 2D bounding box as a thin rectangle."""
        x1, y1, x2, y2 = bbox
        # Dimmed version of the class color
        dim_color = tuple(max(0, c // 2) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), dim_color, 1)

    def _draw_label(
        self,
        frame: np.ndarray,
        pc: ProjectedCuboid,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw class name and distance label above the 2D bbox."""
        x1, y1, x2, y2 = pc.bbox_2d
        distance = pc.cuboid_3d.distance
        class_name = pc.cuboid_3d.class_name.capitalize()
        confidence = pc.cuboid_3d.confidence

        label = f"{class_name} {distance:.1f}m"
        sublabel = f"{confidence:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Measure text
        (tw, th), baseline = cv2.getTextSize(
            label, font, self._label_font_scale, self._label_thickness
        )
        (sw, sh), _ = cv2.getTextSize(
            sublabel, font, self._label_font_scale * 0.7, 1
        )

        # Position above the bbox
        label_y = y1 - 8
        if label_y - th - 6 < 0:
            label_y = y2 + th + 12

        # Draw background rectangle
        bg_x1 = x1
        bg_y1 = label_y - th - 6
        bg_x2 = x1 + max(tw, sw) + 12
        bg_y2 = label_y + sh + 4

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay, self._label_bg_alpha, frame, 1 - self._label_bg_alpha, 0, frame
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y),
            font,
            self._label_font_scale,
            color,
            self._label_thickness,
        )
        cv2.putText(
            frame,
            sublabel,
            (x1 + 4, label_y + sh + 4),
            font,
            self._label_font_scale * 0.7,
            (180, 180, 180),
            1,
        )

    @staticmethod
    def _draw_depth_overlay(
        frame: np.ndarray,
        depth_map: np.ndarray,
        colormap: int,
        alpha: float,
    ) -> np.ndarray:
        """Blend a colorized depth map onto the frame."""
        # Normalize depth to 0-255
        d = depth_map.copy()
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d = np.zeros_like(d, dtype=np.uint8)

        # Resize to frame dimensions if needed
        if d.shape[:2] != frame.shape[:2]:
            d = cv2.resize(d, (frame.shape[1], frame.shape[0]))

        colored_depth = cv2.applyColorMap(d, colormap)
        blended = cv2.addWeighted(frame, 1 - alpha, colored_depth, alpha, 0)
        return blended

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Look up the BGR color for a class name."""
        key = class_name.lower()
        return self._colors.get(key, self._colors.get("default", (200, 200, 200)))

    # ------------------------------------------------------------------
    # Utility — FPS overlay
    # ------------------------------------------------------------------

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw an FPS counter in the top-left corner."""
        label = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            label,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame
