import os
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# Step 1: Prepare the dataset
# Ensure your defect dataset is in the YOLO format
dataset_path = "path/to/defect_dataset"

# Step 2: Define a YAML file for your dataset
yaml_content = f"""
path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 5  # number of classes
names: ['scratch', 'dent', 'crack', 'stain', 'deformation']  # class names
"""

with open("defect_dataset.yaml", "w") as f:
    f.write(yaml_content)

# Step 3: Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Step 4: Fine-tune the model on your defect dataset
results = model.train(
    data='defect_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='defect_detection_model'
)

# Step 5: Perform defect detection on production line
def detect_defects(model_path, source='0', conf_threshold=0.25, save_dir='defect_images'):
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Open video capture (0 for webcam, or provide a video file path or IP camera stream URL)
    cap = cv2.VideoCapture(source)
    
    # Create directory to save defective product images
    os.makedirs(save_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame, conf=conf_threshold)
        
        defects_detected = False
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = box.cls[0].astype(int)
                conf = box.conf[0]
                
                defect_type = model.names[class_id]
                color = (0, 0, 255)  # Red for defects
                label = f"{defect_type}: {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                defects_detected = True
        
        # Display the result
        cv2.imshow("Defect Detection", frame)
        
        # Save image if defects are detected
        if defects_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"defect_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Defect detected! Image saved: {save_path}")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_defects("path/to/trained_defect_model.pt", source='0', conf_threshold=0.25, save_dir='detected_defects')