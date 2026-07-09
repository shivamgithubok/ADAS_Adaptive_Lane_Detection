import cv2
from pathlib import Path
cap = cv2.VideoCapture("test-video_720.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
if ret:
    cv2.imwrite("test_frame_60.jpg", frame)
    print("Frame 60 saved.")
else:
    print("Failed to read frame 60.")
