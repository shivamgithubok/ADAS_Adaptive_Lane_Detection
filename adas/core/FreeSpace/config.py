from dataclasses import dataclass, field
from typing import List

@dataclass
class FreeSpaceConfig:
    """Configuration for Free Space Detection using Depth Anything V2."""
    
    # Model Configuration
    encoder: str = 'vits'  # 'vits', 'vitb', 'vitl', 'vitg'
    input_size: int = 518
    model_filename: str = 'depth_anything_v2_vits.pth'
    repo_dirname: str = 'depth_anything_v2_repo'
    
    # Device configuration
    device: str = "cuda"  # Typically overridden dynamically based on availability
    
    # Geometric Reasoning / Free Space parameters
    # Define a bottom region of interest (ROI) to consider as drivable space.
    # expressed as fractions of the image height/width.
    roi_bottom_start_y: float = 0.6  # Start looking for free space from 60% of the image height down
    
    # Threshold for contiguous region growing or simple depth thresholding
    # Since depth is normalized 0-255, we can use a relative depth variation threshold
    depth_tolerance: float = 20.0
    
    # Median filter size to smooth depth map
    median_blur_size: int = 5
    
    # Geometric pipeline thresholds
    sky_cutoff_ratio: float = 0.35  # Ignore top 35% of the image as sky
    depth_gradient_threshold: float = 15.0  # Threshold for y-axis gradient to detect obstacles
    morph_kernel_size: int = 5  # Kernel size for morphological cleanup
    
    @property
    def model_configs(self) -> dict:
        configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        return configs.get(self.encoder, configs['vits'])
