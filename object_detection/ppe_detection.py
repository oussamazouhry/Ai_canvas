import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Step 1: Prepare the dataset
# Ensure your PPE dataset is in the YOLO format
dataset_path = "path/to/ppe_dataset"

# Step 2: Define a YAML file for your dataset
yaml_content = """
path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 5  # number of classes
names: ['helmet', 'vest', 'gloves', 'boots', 'goggles']  # class names
"""

with open("ppe_dataset.yaml", "w") as f:
    f.write(yaml_content)

# Step 3: Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Step 4: Fine-tune the model on your PPE dataset
results = model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='ppe_detection_model'
)

# Step 5: Perform inference on new images
def detect_ppe(image_path, model_path):
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Perform inference
    results = model(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            class_id = box.cls[0].astype(int)
            conf = box.conf[0]
            
            print(f"Detected {model.names[class_id]} with confidence {conf:.2f} at location {x1},{y1},{x2},{y2}")

    # Visualize results
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_plot = results[0].plot()
    result_plot = cv2.cvtColor(result_plot, cv2.COLOR_RGB2BGR)
    cv2.imshow("PPE Detection Result", result_plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_ppe("path/to/test_image.jpg", "path/to/trained_model.pt")