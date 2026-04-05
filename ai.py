import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # nano model (fastest), or use 'yolov8s.pt', 'yolov8m.pt', etc.

# Open camera (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Run YOLO detection on the frame
    results = model(frame)
    
    # Draw results on the frame
    annotated_frame = results[0].plot()
    
    # Display the frame
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()