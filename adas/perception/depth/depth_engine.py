"""
Depth Anything V2 Engine
========================
Wraps the Depth Anything V2 model for dense monocular depth inference.
Returns a full-resolution depth map (H×W float32 array).

Responsibilities:
    - Load DepthAnythingV2 model weights
    - Preprocess frame and run inference
    - Convert disparity output to pseudo-metric depth
    - Return dense depth map at original resolution

Note:
    Depth Anything V2 (relative variant) outputs **disparity-like** values
    where HIGHER = CLOSER.  This engine converts them to depth
    (HIGHER = FARTHER) and scales to approximate metric range.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger("adas.perception.depth")


class DepthEngine:
    """Wraps Depth Anything V2 (ViT-S/B/L) for dense depth inference."""

    def __init__(
        self,
        model_path: Path,
        encoder: str = "vits",
        input_size: int = 518,
        device: str = "auto",
        repo_path: Path = None,
        near_distance: float = 2.0,
        far_distance: float = 80.0,
    ) -> None:
        """
        Args:
            model_path: Path to the .pth weights file.
            encoder: Encoder variant — 'vits', 'vitb', or 'vitl'.
            input_size: Input resolution for the model (must be multiple of 14).
            device: 'cuda', 'cpu', 'mps', or 'auto' for auto-detection.
            repo_path: Path to the depth_anything_v2_repo directory
                       (needed to import the model architecture).
            near_distance: Approximate nearest distance in meters for scaling.
            far_distance: Approximate farthest distance in meters for scaling.
        """
        self._model_path = Path(model_path)
        self._encoder = encoder
        self._input_size = input_size
        self._device = self._resolve_device(device)
        self._near_distance = near_distance
        self._far_distance = far_distance

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 weights not found: {self._model_path}"
            )

        # Ensure the repo is on sys.path so we can import the model class
        if repo_path is not None:
            repo_str = str(Path(repo_path).resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        self._model = self._load_model()
        logger.info(
            "DepthEngine initialized — encoder=%s, device=%s, input_size=%d, "
            "near=%.1fm, far=%.1fm",
            self._encoder,
            self._device,
            self._input_size,
            self._near_distance,
            self._far_distance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth inference on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3), uint8.

        Returns:
            Depth map as float32 array (H, W).
            Values are in approximate meters (higher = farther).
        """
        with torch.no_grad():
            raw = self._model.infer_image(frame, input_size=self._input_size)

        depth = self._disparity_to_depth(raw.astype(np.float32))
        return depth

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """
        Convert disparity-like output (higher=closer) to metric-ish depth
        (higher=farther) using percentile-based normalization.

        Maps the 2nd-percentile disparity → far_distance
        and the 98th-percentile disparity → near_distance.
        """
        p_low = np.percentile(disp, 2)
        p_high = np.percentile(disp, 98)

        if p_high - p_low < 1e-6:
            # Uniform depth scene — return mid-range
            return np.full_like(disp, (self._near_distance + self._far_distance) / 2.0)

        # Normalize disparity to [0, 1] where 1 = closest
        disp_norm = (disp - p_low) / (p_high - p_low)
        disp_norm = np.clip(disp_norm, 0.0, 1.0)

        # Invert: depth = far when disparity is low, near when high
        # Use inverse mapping for more natural depth distribution
        # depth = far / (1 + (far/near - 1) * disp_norm)
        ratio = self._far_distance / self._near_distance
        depth = self._far_distance / (1.0 + (ratio - 1.0) * disp_norm)

        return depth.astype(np.float32)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Determine the best available compute device."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load and configure the Depth Anything V2 model."""
        from depth_anything_v2.dpt import DepthAnythingV2

        # Encoder → model configuration
        encoder_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        if self._encoder not in encoder_configs:
            raise ValueError(
                f"Unsupported encoder '{self._encoder}'. "
                f"Choose from: {list(encoder_configs.keys())}"
            )

        config = encoder_configs[self._encoder]
        logger.info("Loading Depth Anything V2 (%s) from %s", self._encoder, self._model_path)

        model = DepthAnythingV2(**config)
        state_dict = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(self._device).eval()

        logger.info("Depth Anything V2 model loaded on %s", self._device)
        return model
