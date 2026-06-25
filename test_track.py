import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("test_video.mp4")
for i in range(10):
    ret, frame = cap.read()
    if not ret: break
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        print(f"Frame {i}: IDs: {boxes.id}")
