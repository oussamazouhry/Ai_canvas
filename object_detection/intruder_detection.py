import os
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Step 1: Prepare the dataset (intruder dataset must be in YOLO format)
dataset_path = "path/to/intruder_dataset"

# Step 2: Define a YAML file for your dataset
yaml_content = """
path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['person', 'intruder']  # class names
"""

with open("intruder_dataset.yaml", "w") as f:
    f.write(yaml_content)

# Step 3: Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Step 4: Fine-tune the model on your intruder dataset
results = model.train(
    data='intruder_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='intruder_detection_model'
)

# Step 5: Perform real-time intruder detection
def detect_intruder(model_path, conf_threshold=0.5, iou_threshold=0.5):
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Open video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame, conf=conf_threshold, iou=iou_threshold)
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = box.cls[0].astype(int)
                conf = box.conf[0]
                
                if model.names[class_id] == 'intruder':
                    color = (0, 0, 255)  # Red for intruders
                    label = f"Intruder: {conf:.2f}"
                else:
                    color = (0, 255, 0)  # Green for regular persons
                    label = f"Person: {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display the result
        cv2.imshow("Intruder Detection", frame)
        
        # Break the loop if 'q' is