import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

from .config import FreeSpaceConfig
from .utils import find_project_root, normalize_depth

logger = logging.getLogger(__name__)

class DepthEngine:
    """
    Handles initialization and inference for the Depth Anything V2 model.
    """
    def __init__(self, config: FreeSpaceConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine project root and add repo to sys.path
        current_dir = Path(__file__).parent
        self.project_root = find_project_root(current_dir, self.config.repo_dirname)
        
        self.repo_path = self.project_root / self.config.repo_dirname
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Depth Anything V2 repo not found at {self.repo_path}")
            
        # Dynamically add the repo to sys.path so we can import DepthAnythingV2
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
            
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(f"Failed to import DepthAnythingV2 from {self.repo_path}. Error: {e}")
            
        self.model = DepthAnythingV2(**self.config.model_configs)
        
        model_path = self.project_root / self.config.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        logger.info(f"Loading Depth Anything V2 from {model_path} onto {self.device}...")
        self.model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        logger.info("DepthEngine initialized successfully.")

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Runs dense depth inference on an RGB image.
        
        Args:
            image (np.ndarray): HxWxC RGB image.
            
        Returns:
            np.ndarray: HxW normalized uint8 depth map (0-255).
        """
        with torch.no_grad():
            depth = self.model.infer_image(image, self.config.input_size)
            
        # Normalize to 0-255 uint8
        depth_norm = normalize_depth(depth)
        return depth_norm
