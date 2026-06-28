import os
from pathlib import Path

# Project Roots
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolo11n-seg.pt"

# Classes to keep (COCO indices: 2=car, 3=motorcycle, 5=bus, 7=truck)
TARGET_CLASSES = [2, 3, 5, 7]

# Target Class Names Mapping
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Detection thresholds
CONF_THRESHOLD = 0.5

# Tracker thresholds
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 2

# Validation Thresholds
MIN_BBOX_HEIGHT = 15
MIN_BBOX_AREA = 200
MIN_MASK_AREA = 100
MIN_GROUND_PIXELS = 10
MIN_CONTOUR_SIZE = 5

# Ground Contact Region config
STRIP_RATIO = 0.10
MIN_STRIP_PIXELS = 4
MAX_STRIP_PIXELS = 20

# Quality thresholds
QUALITY_THRESHOLDS = {
    'mask_area_weight': 20,
    'ground_pixels_weight': 30,
    'ground_width_weight': 30,
    'contact_geometry_weight': 20
}

DRAW_STRIP = True
DRAW_CONTACT_POINTS = True
DEBUG_GROUND_CONTACT = True

# Visualization Settings
COLOR_MASK = (0, 255, 0)         # Green
COLOR_BBOX = (255, 255, 255)     # White
COLOR_TEXT = (255, 255, 255)
COLOR_STRIP = (0, 255, 255)      # Yellow for line and strip outline
COLOR_L_CONTACT = (0, 0, 255)    # Red (BGR)
COLOR_R_CONTACT = (255, 0, 0)    # Blue (BGR)
COLOR_M_CONTACT = (255, 255, 0)  # Cyan (BGR)
COLOR_GROUND_LINE = (0, 255, 255)# Yellow

ALPHA_MASK = 0.4
ALPHA_STRIP = 0.3
THICKNESS_BBOX = 2
THICKNESS_STRIP = 2
THICKNESS_TEXT = 2
RADIUS_CONTACT = 4
FONT_SCALE_LABEL = 0.5
FONT_SCALE_FPS = 1.0
