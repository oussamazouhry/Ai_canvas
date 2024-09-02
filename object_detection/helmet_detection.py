import os
from ultralytics import YOLO
import cv2
import numpy as np

# Step 1: Prepare the dataset
# Ensure your helmet dataset is in the YOLO format
dataset_path = "path/to/helmet_dataset"

# Step 2: Define a YAML file for your dataset
yaml_content = f"""
path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['person', 'helmet']  # class names
"""

with open("helmet_dataset.yaml", "w") as f:
    f.write(yaml_content)

# Step 3: Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Step 4: Fine-tune the model on your helmet dataset
results = model.train(
    data='helmet_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='helmet_detection_model'
)

# Step 5: Perform helmet detection
def detect_helmets(model_path, source='0', conf_threshold=0.25):
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Open video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame, conf=conf_threshold)
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = box.cls[0].astype(int)
                conf = box.conf[0]
                
                if model.names[class_id] == 'person':
                    color = (0, 255, 0)  # Green for person
                    label = f"Person: {conf:.2f}"
                elif model.names[class_id] == 'helmet':
                    color = (0, 255, 255)  # Yellow for helmet
                    label = f"Helmet: {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display the result
        cv2.imshow("Helmet Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_helmets("path/to/trained_helmet_model.pt", source='0', conf_threshold=0.25)

# Additional function for helmet violation detection
def detect_helmet_violations(model_path, source='0', conf_threshold=0.25):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=conf_threshold)
        
        persons = []
        helmets = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = box.cls[0].astype(int)
                conf = box.conf[0]
                
                if model.names[class_id] == 'person':
                    persons.append((x1, y1, x2, y2))
                elif model.names[class_id] == 'helmet':
                    helmets.append((x1, y1, x2, y2))
        
        for person in persons:
            has_helmet = False
            for helmet in helmets:
                if (helmet[0] > person[0] and helmet[1] > person[1] and
                    helmet[2] < person[2] and helmet[3] < person[3]):
                    has_helmet = True
                    break
            
            color = (0, 255, 0) if has_helmet else (0, 0, 255)  # Green if has helmet, Red if not
            label = "Safe" if has_helmet else "Violation"
            cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), color, 2)
            cv2.putText(frame, label, (person[0], person[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Helmet Violation Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage of helmet violation detection
detect_helmet_violations("path/to/trained_helmet_model.pt", source='0', conf_threshold=0.25)