import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "configs"

class SettingsManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance._load_all()
        return cls._instance
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        filepath = CONFIG_DIR / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
            
    def _load_all(self):
        self.camera = self._load_yaml("camera.yaml")
        self.vehicle = self._load_yaml("vehicle.yaml")
        self.tracker = self._load_yaml("tracker.yaml")
        self.lane = self._load_yaml("lane.yaml")
        self.visualization = self._load_yaml("visualization.yaml")
        
        # Add class names mapping
        self.class_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    # --- Backward Compatibility Properties ---
    @property
    def MODEL_PATH(self):
        return BASE_DIR / "yolo11n-seg.pt"
        
    @property
    def TARGET_CLASSES(self):
        return self.vehicle.get('target_classes', [2, 3, 5, 7])
        
    @property
    def CLASS_NAMES(self):
        return {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        
    @property
    def CONF_THRESHOLD(self):
        return self.vehicle.get('conf_threshold', 0.5)
        
    @property
    def TRACK_MAX_AGE(self):
        return self.tracker.get('max_age', 30)
        
    @property
    def TRACK_MIN_HITS(self):
        return self.tracker.get('min_hits', 2)
        
    @property
    def MIN_BBOX_HEIGHT(self):
        return self.vehicle.get('min_bbox_height', 15)
        
    @property
    def MIN_BBOX_AREA(self):
        return self.vehicle.get('min_bbox_area', 200)
        
    @property
    def MIN_MASK_AREA(self):
        return self.vehicle.get('min_mask_area', 100)
        
    @property
    def MIN_GROUND_PIXELS(self):
        return self.vehicle.get('min_ground_pixels', 10)
        
    @property
    def MIN_CONTOUR_SIZE(self):
        return self.vehicle.get('min_contour_size', 5)
        
    @property
    def STRIP_RATIO(self):
        return self.vehicle.get('strip_ratio', 0.10)
        
    @property
    def MIN_STRIP_PIXELS(self):
        return self.vehicle.get('min_strip_pixels', 4)
        
    @property
    def MAX_STRIP_PIXELS(self):
        return self.vehicle.get('max_strip_pixels', 20)
        
    @property
    def QUALITY_THRESHOLDS(self):
        return self.vehicle.get('quality_thresholds', {})
        
    # Visualization Backward Compatibility
    @property
    def COLOR_MASK(self): return tuple(self.visualization.get('colors', {}).get('mask', (0, 255, 0)))
    @property
    def COLOR_BBOX(self): return tuple(self.visualization.get('colors', {}).get('bbox', (255, 255, 255)))
    @property
    def COLOR_TEXT(self): return tuple(self.visualization.get('colors', {}).get('text', (255, 255, 255)))
    @property
    def COLOR_STRIP(self): return tuple(self.visualization.get('colors', {}).get('strip', (0, 255, 255)))
    @property
    def COLOR_L_CONTACT(self): return tuple(self.visualization.get('colors', {}).get('left_contact', (0, 0, 255)))
    @property
    def COLOR_R_CONTACT(self): return tuple(self.visualization.get('colors', {}).get('right_contact', (255, 0, 0)))
    @property
    def COLOR_M_CONTACT(self): return tuple(self.visualization.get('colors', {}).get('center_contact', (255, 255, 0)))
    @property
    def COLOR_GROUND_LINE(self): return tuple(self.visualization.get('colors', {}).get('ground_line', (0, 255, 255)))
    
    @property
    def ALPHA_MASK(self): return self.visualization.get('alpha', {}).get('mask', 0.4)
    @property
    def ALPHA_STRIP(self): return self.visualization.get('alpha', {}).get('strip', 0.3)
    @property
    def THICKNESS_BBOX(self): return self.visualization.get('thickness', {}).get('bbox', 2)
    @property
    def THICKNESS_STRIP(self): return self.visualization.get('thickness', {}).get('strip', 2)
    @property
    def THICKNESS_TEXT(self): return self.visualization.get('thickness', {}).get('text', 2)
    @property
    def RADIUS_CONTACT(self): return self.visualization.get('radius', {}).get('contact', 4)
    @property
    def FONT_SCALE_LABEL(self): return self.visualization.get('font_scale', {}).get('label', 0.5)
    @property
    def FONT_SCALE_FPS(self): return self.visualization.get('font_scale', {}).get('fps', 1.0)
    
    @property
    def DRAW_STRIP(self): return self.visualization.get('debug', {}).get('draw_strip', True)
    @property
    def DRAW_CONTACT_POINTS(self): return self.visualization.get('debug', {}).get('draw_contact_points', True)
    @property
    def DEBUG_GROUND_CONTACT(self): return self.visualization.get('debug', {}).get('ground_contact', True)


# Global singleton
config = SettingsManager()
