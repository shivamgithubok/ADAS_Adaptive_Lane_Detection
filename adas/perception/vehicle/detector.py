from ultralytics import YOLO
from adas.config.settings import config

class Detector:
    def __init__(self):
        # Load the YOLO segmentation model
        self.model = YOLO(config.MODEL_PATH)

    def detect_and_track(self, frame, persist=True):
        """
        Runs YOLO11 segmentation model with built-in tracking (ByteTrack/BoT-SORT).
        Filters for target classes.
        """
        results = self.model.track(
            source=frame, 
            persist=persist, 
            classes=config.TARGET_CLASSES,
            conf=config.CONF_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml"  # Using default ByteTrack config from ultralytics
        )
        return results[0] if len(results) > 0 else None
