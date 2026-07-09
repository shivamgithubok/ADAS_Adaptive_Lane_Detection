"""
YOLO Detection Engine
=====================
Wraps ultralytics YOLO for pure 2D object detection.
Returns Detection2D dataclasses — no segmentation, no tracking.

Responsibilities:
    - Load YOLO model
    - Run inference on a single RGB frame
    - Filter by target classes and confidence
    - Return clean Detection2D list
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from adas.common.datatypes import Detection2D

logger = logging.getLogger("adas.perception.detection")


class YoloEngine:
    """Lightweight YOLO detection wrapper for the 3D perception pipeline."""

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        class_id_map: Optional[dict] = None,
    ) -> None:
        """
        Args:
            model_path: Path to the YOLO .pt weights file.
            conf_threshold: Minimum confidence to keep a detection.
            target_classes: COCO class IDs to detect. None = all classes.
            class_id_map: Mapping from COCO class ID to human-readable name.
        """
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self._model_path}")

        self._conf_threshold = conf_threshold
        self._target_classes = target_classes
        self._class_id_map = class_id_map or {}

        logger.info("Loading YOLO model from %s", self._model_path)
        self._model = YOLO(str(self._model_path))
        logger.info("YOLO model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        """
        Run YOLO detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection2D objects, sorted by confidence (desc).
        """
        results = self._model(
            source=frame,
            conf=self._conf_threshold,
            classes=self._target_classes,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        return self._parse_results(results[0])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_results(self, result) -> List[Detection2D]:
        """Extract Detection2D objects from a single YOLO result."""
        detections: List[Detection2D] = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            class_name = self._class_id_map.get(
                class_id, result.names.get(class_id, f"class_{class_id}")
            )

            detections.append(
                Detection2D(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
