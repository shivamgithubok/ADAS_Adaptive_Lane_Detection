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
├── app.py              # Flask server and MJPEG stream handlers
├── lane_detection.py   # Core pipeline (Week 3 Research Build)
├── templates/
│   └── index.html      # Responsive frontend (Inter font, Dark UI)
├── requirements.txt    # Basic dependencies
├── pyproject.toml      # Detailed project metadata and dependencies
└── yolov8n.pt          # YOLO weights (auto-downloaded on first run)
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

   # Install dependencies
   pip install -e .
   ```

   > [!NOTE]
   > The first time you run the app, it will automatically download the YOLOv8 weights (`yolov8n.pt` or `yolo11n-seg.pt`) for vehicle detection.

---

## 🚦 Usage

### Run the Web App
Start the Flask server:
```bash
python app.py
```
Then open your browser at: `http://localhost:7860`

### Optional CLI Preview
You can also run a local OpenCV window preview:
```bash
python lane_detection.py path/to/video.mp4 --debug
```

### Run Individual Phases
To run the specific phases of the pipeline directly from the command line:

- **Phase 00 (Profiling)**:
  ```bash
  python phase_00_profiling.py
  ```
- **Phase 01 (Perception)**:
  ```bash
  python phase_01_perception/run.py   --video "path_of viedo"
  ```
- **Phase 02 (Geometry)**:
  ```bash
  python phase_02_geometry/run.py  --video "path_of viedo"
  ```
- **Phase 03 (Lane Detection)**:
  ```bash
  python phase_03_lane/run.py --video "path_of viedo"
  ```

---

## 🔍 How It Works (Week 3 Pipeline)

The processing pipeline in [`lane_detection.py`](lane_detection.py) follows these steps:

1. **Vehicle Detection**: YOLOv8 identifies cars, trucks, and motorbikes.
2. **HLS Masking**: Extracts white and yellow pixels from the frame.
3. **Vehicle Subtraction**: Subtracts vehicle regions from the HLS mask to "clean" the road area.
4. **Dynamic ROI**: Sets the ROI height based on the "lead car" position or auto-calibration.
5. **Edge Detection**: Applies Otsu thresholding and Canny edge detection.
6. **Curve Fitting**: Uses RANSAC and Hough Transforms to find lane lines.
7. **Smoothing**: Applies EMA and temporal buffers to prevent flickering.

---

## 📜 Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics) for object detection.
- **OpenCV** for computer vision primitives.
- Inspired by the Udacity Self-Driving Car Nanodegree and Automatic Addison's lane detection guides.

#   A D A S _ A d a p t i v e _ L a n e _ D e t e c t i o n  