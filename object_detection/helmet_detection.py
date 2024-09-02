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

# Step 5: Perform helmet detection on images
def detect_helmets_in_image(model_path, image_path, conf_threshold=0.25, save_path=None):
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Load and process the image
    image = Image.open(image_path)
    results = model(image, conf=conf_threshold)[0]
    
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process and draw results
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        if model.names[class_id] == 'person':
            color = (0, 255, 0)  # Green for person
            label = f"Person: {conf:.2f}"
        elif model.names[class_id] == 'helmet':
            color = (0, 255, 255)  # Yellow for helmet
            label = f"Helmet: {conf:.2f}"
        
        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(opencv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display or save the result
    if save_path:
        cv2.imwrite(save_path, opencv_image)
        print(f"Processed image saved to {save_path}")
    else:
        cv2.imshow("Helmet Detection", opencv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage for a single image
detect_helmets_in_image("path/to/trained_helmet_model.pt", "path/to/test_image.jpg", save_path="output_image.jpg")

# Function to process a batch of images
def process_image_batch(model_path, input_folder, output_folder, conf_threshold=0.25):
    model = YOLO(model_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            
            detect_helmets_in_image(model_path, input_path, conf_threshold, save_path=output_path)
    
    print(f"Processed all images. Results saved in {output_folder}")

# Example usage for batch processing
process_image_batch("path/to/trained_helmet_model.pt", "path/to/input_images", "path/to/output_images")