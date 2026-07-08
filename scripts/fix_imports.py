import os
import re
from pathlib import Path

BASE_DIR = Path('/home/elevatics/Projects/ADAS_Adaptive_Lane_Detection/adas')

REPLACEMENTS = {
    r"from config import": "from adas.config.settings import",
    r"import config": "from adas.config.settings import config",
    r"import config as p1_config": "from adas.config.settings import config as p1_config",
    
    r"from detector import": "from adas.perception.vehicle.detector import",
    r"from segmentation import": "from adas.perception.vehicle.segmentation import",
    r"from contact_region import": "from adas.perception.vehicle.contact_region import",
    r"from occlusion_manager import": "from adas.perception.vehicle.occlusion_manager import",
    
    r"import lane_detection_v0": "import adas.perception.lane.lane_detection_v0",
    r"from lane_geometry import": "from adas.perception.lane.lane_geometry import",
    r"from lane_assignment import": "from adas.perception.lane.lane_assignment import",
    r"from operational_zone import": "from adas.perception.lane.operational_zone import",
    
    r"from homography import": "from adas.core.geometry.homography import",
    r"from metric_scaling import": "from adas.core.geometry.metric_scaling import",
    r"from footprint_stabilizer import": "from adas.core.geometry.footprint_stabilizer import",
    r"from perspective_compensator import": "from adas.core.geometry.perspective_compensator import",
    
    r"from projector import": "from adas.core.projection.projector import",
    r"from lane_projector import": "from adas.core.projection.lane_projector import",
    
    r"from tracker import": "from adas.core.tracking.tracker import",
    
    r"from utils import": "from adas.common.utils import",
    
    r"from visualization import": "from adas.visualization.vehicle_vis import",  # Approximation
    
    r"sys.path.append\(str\(base_dir / 'phase_01_perception'\)\)": "",
    r"sys.path.append\(str\(base_dir / 'phase_02_geometry'\)\)": "",
    r"sys.path.append\(str\(base_dir\)\)": "sys.path.append(str(base_dir.parent))"
}

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith('.py'):
            filepath = Path(root) / file
            with open(filepath, 'r') as f:
                content = f.read()
            
            for pattern, repl in REPLACEMENTS.items():
                content = re.sub(pattern, repl, content)
                
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated {filepath}")
