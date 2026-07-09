#!/usr/bin/env python3
"""
run_perception_3d.py — Monocular 3D Perception Demo
=====================================================
Standalone CLI runner that demonstrates the full YOLO + Depth Anything V2
fusion pipeline for monocular 3D object perception.

Usage:
    python scripts/run_perception_3d.py --video test-video_720.mp4
    python scripts/run_perception_3d.py --video test-video_720.mp4 --output outputs/demo_3d.mp4
    python scripts/run_perception_3d.py --image some_frame.jpg
    python scripts/run_perception_3d.py --video test-video_720.mp4 --no-display --output outputs/demo.mp4
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adas.common.datatypes import CameraIntrinsics, Detection2D, ProjectedCuboid
from adas.perception.detection.yolo_engine import YoloEngine
from adas.perception.depth.depth_engine import DepthEngine
from adas.perception.fusion.object_depth import ObjectDepthEstimator
from adas.perception.fusion.cuboid_generator import CuboidGenerator
from adas.perception.fusion.projection import CuboidProjector
from adas.visualization.bbox3d_vis import BBox3DVisualizer

logger = logging.getLogger("adas.perception_3d")


# ============================================================
# Configuration Loader
# ============================================================

def load_config(config_path: Path) -> dict:
    """Load the perception_3d.yaml config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_camera_intrinsics(config_dir: Path, frame_width: int = 0, frame_height: int = 0) -> CameraIntrinsics:
    """
    Load camera intrinsics from camera.yaml.
    If frame dimensions differ from calibration dimensions, scale intrinsics.
    """
    cam_path = config_dir / "camera.yaml"
    if not cam_path.exists():
        logger.warning("camera.yaml not found, using defaults")
        return CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    with open(cam_path, "r") as f:
        cam = yaml.safe_load(f)

    fx = cam.get("fx", 800.0)
    fy = cam.get("fy", 800.0)
    cx = cam.get("cx", 640.0)
    cy = cam.get("cy", 360.0)
    cal_w = cam.get("calibration_width", 1280)
    cal_h = cam.get("calibration_height", 720)

    # Scale intrinsics if the actual frame size differs from calibration
    if frame_width > 0 and frame_height > 0 and (frame_width != cal_w or frame_height != cal_h):
        sx = frame_width / cal_w
        sy = frame_height / cal_h
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
        logger.info(
            "Scaled intrinsics from %dx%d → %dx%d (sx=%.3f, sy=%.3f)",
            cal_w, cal_h, frame_width, frame_height, sx, sy,
        )

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


# ============================================================
# Pipeline Orchestrator
# ============================================================

class Perception3DPipeline:
    """Orchestrates the full YOLO + Depth → 3D cuboid pipeline."""

    def __init__(self, config: dict, intrinsics: CameraIntrinsics, project_root: Path):
        cfg_yolo = config["yolo"]
        cfg_depth = config["depth"]
        cfg_sampling = config.get("depth_sampling", {})
        cfg_viz = config.get("visualization", {})
        class_id_map = config.get("class_id_map", {})
        # Convert string keys from YAML to int
        class_id_map = {int(k): v for k, v in class_id_map.items()}

        # --- YOLO Engine ---
        yolo_path = project_root / cfg_yolo["model_path"]
        self.yolo = YoloEngine(
            model_path=yolo_path,
            conf_threshold=cfg_yolo.get("conf_threshold", 0.45),
            target_classes=cfg_yolo.get("target_classes"),
            class_id_map=class_id_map,
        )

        # --- Depth Engine ---
        depth_path = project_root / cfg_depth["model_path"]
        repo_path = project_root / "depth_anything_v2_repo"
        self.depth = DepthEngine(
            model_path=depth_path,
            encoder=cfg_depth.get("encoder", "vits"),
            input_size=cfg_depth.get("input_size", 518),
            repo_path=repo_path,
            near_distance=cfg_depth.get("near_distance", 2.0),
            far_distance=cfg_depth.get("far_distance", 80.0),
        )

        # --- Fusion Modules ---
        self.depth_estimator = ObjectDepthEstimator(
            roi_shrink=cfg_sampling.get("roi_shrink", 0.15),
            min_valid_pixels=cfg_sampling.get("min_valid_pixels", 10),
        )

        class_priors = config.get("class_priors", {})
        self.cuboid_gen = CuboidGenerator(
            class_priors=class_priors,
            intrinsics=intrinsics,
        )

        self.projector = CuboidProjector(intrinsics=intrinsics)

        # --- Visualization ---
        viz_colors_raw = cfg_viz.get("colors", {})
        viz_colors = {}
        for k, v in viz_colors_raw.items():
            if isinstance(v, list) and len(v) == 3:
                viz_colors[k] = tuple(v)
        cuboid_cfg = cfg_viz.get("cuboid", {})
        label_cfg = cfg_viz.get("label", {})

        self.visualizer = BBox3DVisualizer(
            color_palette=viz_colors if viz_colors else None,
            front_alpha=cuboid_cfg.get("front_alpha", 0.25),
            front_thickness=cuboid_cfg.get("front_thickness", 2),
            rear_thickness=cuboid_cfg.get("rear_thickness", 1),
            edge_thickness=cuboid_cfg.get("edge_thickness", 1),
            label_font_scale=label_cfg.get("font_scale", 0.55),
            label_thickness=label_cfg.get("thickness", 2),
            label_bg_alpha=label_cfg.get("background_alpha", 0.6),
        )

        # Depth overlay settings
        depth_overlay_cfg = cfg_viz.get("depth_overlay", {})
        self.show_depth_overlay = depth_overlay_cfg.get("enabled", True)
        self.depth_overlay_alpha = depth_overlay_cfg.get("alpha", 0.15)
        self.depth_colormap = depth_overlay_cfg.get("colormap", cv2.COLORMAP_JET)

        # Threading pool for parallel YOLO + Depth
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._intrinsics = intrinsics
        logger.info("Perception3DPipeline initialized")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full pipeline on a single BGR frame.
        YOLO and Depth Anything run in parallel for better throughput.

        Returns:
            Annotated frame with 3D cuboids, labels, and optional depth overlay.
        """
        # Step 1 & 2: Run YOLO and Depth in PARALLEL
        yolo_future = self._executor.submit(self.yolo.detect, frame)
        depth_future = self._executor.submit(self.depth.infer, frame)

        detections = yolo_future.result()
        depth_map = depth_future.result()

        # Steps 3-7: Fusion for each detection
        projected_cuboids = self._fuse_detections(detections, depth_map)

        # Step 7: Visualization
        output = self.visualizer.draw_all(
            frame,
            projected_cuboids,
            depth_map=depth_map,
            depth_colormap=self.depth_colormap,
            depth_alpha=self.depth_overlay_alpha,
            show_depth_overlay=self.show_depth_overlay,
        )

        return output

    def _fuse_detections(
        self,
        detections: List[Detection2D],
        depth_map: np.ndarray,
    ) -> List[ProjectedCuboid]:
        """Run fusion steps (depth sampling → cuboid → projection) per detection."""
        projected_cuboids = []

        for det in detections:
            # Step 3: Depth sampling
            depth_est = self.depth_estimator.estimate(depth_map, det.bbox)
            if depth_est is None:
                continue

            # Step 4-5: Cuboid generation (3D center + dimensions)
            cuboid = self.cuboid_gen.generate(det, depth_est)
            if cuboid is None:
                continue

            # Step 6: Project 3D → 2D
            projected = self.projector.project(cuboid, det, depth_est)
            projected_cuboids.append(projected)

        return projected_cuboids


# ============================================================
# CLI Runner
# ============================================================

def run_video(pipeline: Perception3DPipeline, video_path: Path, output_path: Path = None, display: bool = True):
    """Process a video file through the 3D perception pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_video, (width, height))
        logger.info("Writing output to %s", output_path)

    logger.info(
        "Processing video: %s (%dx%d, %.1f fps, %d frames)",
        video_path.name, width, height, fps_video, total_frames,
    )

    frame_idx = 0
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            output = pipeline.process_frame(frame)
            t1 = time.perf_counter()

            # FPS calculation (exponential moving average)
            instant_fps = 1.0 / max(t1 - t0, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

            # Draw FPS
            BBox3DVisualizer.draw_fps(output, fps_smooth)

            # Progress
            if frame_idx % 30 == 0:
                logger.info(
                    "Frame %d/%d — FPS: %.1f — Latency: %.1f ms",
                    frame_idx, total_frames, fps_smooth, (t1 - t0) * 1000,
                )

            if writer is not None:
                writer.write(output)

            if display:
                cv2.imshow("ADAS 3D Perception", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User requested exit")
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    logger.info("Processing complete — %d frames processed", frame_idx)


def run_image(pipeline: Perception3DPipeline, image_path: Path, output_path: Path = None, display: bool = True):
    """Process a single image through the 3D perception pipeline."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Cannot read image: %s", image_path)
        return

    t0 = time.perf_counter()
    output = pipeline.process_frame(frame)
    t1 = time.perf_counter()

    logger.info("Processed in %.1f ms", (t1 - t0) * 1000)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)
        logger.info("Saved to %s", output_path)

    if display:
        cv2.imshow("ADAS 3D Perception", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monocular 3D Perception — YOLO + Depth Anything V2 Fusion"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input video file")
    group.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, default=None, help="Path to output file")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to perception_3d.yaml (default: configs/perception_3d.yaml)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV display window (headless mode)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "perception_3d.yaml"
    config = load_config(config_path)

    # Determine frame size for intrinsic scaling
    frame_w, frame_h = 0, 0
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    elif args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        img = cv2.imread(str(image_path))
        if img is not None:
            frame_h, frame_w = img.shape[:2]

    intrinsics = load_camera_intrinsics(PROJECT_ROOT / "configs", frame_w, frame_h)

    # Build pipeline
    pipeline = Perception3DPipeline(config, intrinsics, PROJECT_ROOT)

    # Run
    display = not args.no_display
    output_path = Path(args.output) if args.output else None

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        run_video(pipeline, video_path, output_path, display)
    else:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        run_image(pipeline, image_path, output_path, display)


if __name__ == "__main__":
    main()
