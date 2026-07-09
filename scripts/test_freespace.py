import cv2
import sys
from pathlib import Path
import logging

# Add project root to sys.path so we can import adas
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from adas.core.FreeSpace import FreeSpaceDetector, FreeSpaceConfig
from adas.visualization.freespace_vis import FreeSpaceVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_freespace(video_path: str):
    logger.info(f"Extracting first frame from {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Failed to read video frame.")
        return
        
    logger.info("Initializing Free Space Detector...")
    config = FreeSpaceConfig()
    detector = FreeSpaceDetector(config)
    visualizer = FreeSpaceVisualizer()
    
    logger.info("Running detection...")
    result = detector.detect(frame)
    
    logger.info("Saving visualization...")
    visualizer.save_visualization(frame, result, 'test_freespace_output.jpg')
    logger.info("Test completed.")

if __name__ == "__main__":
    video = str(project_root / "testing_video" / "test_1.mp4")
    if len(sys.argv) > 1:
        video = sys.argv[1]
    test_freespace(video)
