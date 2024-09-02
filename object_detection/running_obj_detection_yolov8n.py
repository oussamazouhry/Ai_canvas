import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
# Benchmark on GPU
# import supervision as sv


def finetuneyolo(data_set):
    # Load the pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Prepare your custom dataset
    # Your dataset should be in the COCO format, with images and annotations
    train_dataset = 'path/to/your/train/dataset'
    val_dataset = 'path/to/your/validation/dataset'

    # Finetune the model on the custom dataset
    model.finetune(data=train_dataset, epochs=50, batch=16, imgsz=640, freeze=['backbone'])

    # Evaluate the finetuned model on the validation dataset
    results = model.val(data=val_dataset)

    # Save the finetuned model
    model.save('yolov8n_finetuned.pt')

    
if __name__=="__main__":
    benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

    #load the model
    model = YOLO("yolov8n.pt")
    model.info()

    #load the impage 
    # image_path = "zidane.jpg"
    image_path = "bus.jpg" 


    # Inference
    results = model(image_path)
    image = results[0].plot()
    
    # plot the results
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, img_rgb)
    # cv2.imshow('Detection Results', img_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def