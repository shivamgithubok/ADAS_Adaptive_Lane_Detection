# Lane Detection — Live Analyzer 🛣️

🚗 **Intelligent Road Perception** | 🛡️ **Vehicle Masking** | ⚡ **Live MJPEG Streaming**

A modern lane detection system built with **Python**, **OpenCV**, and **Flask**. This version (Week 3 Research Build) integrates **YOLOv8** for real-time vehicle detection to improve lane tracking stability by masking out cars and dynamically adjusting the Region of Interest (ROI).

![Lane Detection Preview](test_video.mp4)

---

## 🚀 Key Features

- **Flask Backend & Modern UI**: A sleek, dark-themed dashboard for video uploading and real-time analysis.
- **MJPEG Streaming**: Three concurrent live streams:
  - **Final Output**: Lane overlay with vehicle bounding boxes.
  - **Canny Edges**: Visualizing the thresholded edge detection.
  - **ROI Mask**: Showing the dynamic perspective and ROI cropping.
- **YOLOv8 Car Masking**: Automatically detects vehicles and removes them from the lane mask to prevent false positives from car reflections or textures.
- **Dynamic ROI**: The ROI top boundary adjusts automatically based on the distance to the nearest lead vehicle.
- **Auto-Calibration [H]**: Uses EMA (Exponential Moving Average) cross-tracking to lock the horizon and ROI geometry after a few frames.
- **Advanced Filtering**: Combines HLS thresholding, Otsu Canny, and RANSAC curve fitting for robust detection.

---

## 📂 Project Structure

```text
.
├── adas/
│   ├── common/         # Shared utilities and dataclasses
│   ├── config/         # SettingsManager for loading YAML configs
│   ├── core/           # Geometry, projection, and tracking modules
│   ├── perception/     # Vehicle and lane detection modules
│   ├── pipelines/      # Execution pipelines (Vehicle, Geometry, Lane)
│   └── visualization/  # Consolidated rendering engines
├── configs/            # YAML configuration files
├── scripts/            # Build/refactor utility scripts
├── archive/            # Legacy experimental phases
├── requirements.txt    # Dependencies
└── app.py              # Flask server and MJPEG stream handlers
```

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.11+**
- **pip** or **uv**

### Installation

1. **Clone the repository** (or navigate to the project directory).
2. **Set up the virtual environment & install dependencies**:

   We recommend using `uv` for fast dependency management, but standard `pip` works perfectly too.

   **Using `uv` (Recommended):**
   ```bash
   # Sync dependencies and create a virtual environment
   uv sync
   # Activate the virtual environment
   source .venv/bin/activate
   ```

   **Using standard `pip`:**
   ```bash
   # Create a virtual environment
   python -m venv .venv
   # Activate it (Linux/macOS)
   source .venv/bin/activate
   # Or on Windows: .venv\Scripts\activate

   # [Optional] Install GPU-specific PyTorch packages for NVIDIA GPUs:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install the required dependencies
   pip install -r requirements.txt
   ```

   > [!NOTE]
   > The first time you run the app, it will automatically download the YOLOv8 weights (`yolov8n.pt` or `yolo11n-seg.pt`) for vehicle detection.

---

## 🚦 Usage

### 🏃‍♂️ Run the Code
To start the lane detection server, run the following command:
```bash
python app.py
```
Then open your browser at: `http://localhost:7860`

### Run Individual Pipelines
To run the specific pipelines directly from the command line:

- **Vehicle Perception Pipeline**:
  ```bash
  python3 adas/pipelines/vehicle_pipeline.py --video adas/test-video_480.mp4
  ```
- **Geometry Pipeline**:
  ```bash
  python3 adas/pipelines/geometry_pipeline.py --video adas/test-video_480.mp4 --estimator smoke
  ```
- **Lane Pipeline**:
  ```bash
  python3 adas/pipelines/lane_pipeline.py --video adas/test-video_480.mp4
  ```

---

## 🔍 How It Works (Production Architecture)

The system has been refactored into a domain-driven architecture for production scalability:

1. **Vehicle Perception (`adas/perception/vehicle`)**: YOLO-based vehicle detection, instance segmentation, and ground contact point extraction.
2. **Lane Perception (`adas/perception/lane`)**: HLS masking, Otsu Canny edge detection, RANSAC curve fitting, and Operational Detection Zone (ODZ) filtering.
3. **Core Geometry (`adas/core/geometry`)**: Homography projection, monocular 3D dimension estimation (SMOKE/RTM3D), footprint stabilization, and metric scaling.
4. **Core Tracking (`adas/core/tracking`)**: DeepSORT/ByteTrack algorithms for temporal consistency and smoothing.
5. **Visualization (`adas/visualization`)**: Decoupled rendering layers drawing structured data models using pre-configured styling tokens.

---

## 📜 Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics) for object detection.
- **OpenCV** for computer vision primitives.
- Inspired by the Udacity Self-Driving Car Nanodegree and Automatic Addison's lane detection guides.

#   A D A S _ A d a p t i v e _ L a n e _ D e t e c t i o n  