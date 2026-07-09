import os
from pathlib import Path
import cv2
import numpy as np

def find_project_root(current_dir: Path, marker_dir: str = 'depth_anything_v2_repo') -> Path:
    """
    Search upwards from current_dir to find the project root 
    which contains the marker_dir.
    """
    path = current_dir.resolve()
    for _ in range(5):  # Limit search depth
        if (path / marker_dir).is_dir():
            return path
        path = path.parent
    
    # Fallback to current working directory if not found
    return Path(os.getcwd())

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Normalize depth map to 0-255 uint8 range.
    """
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    return depth.astype(np.uint8)

def apply_color_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Apply SPECTRAL colormap to a 0-255 uint8 depth map.
    Close objects appear red, far objects appear blue.
    """
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_colored = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth_colored
