from ultralytics import YOLO
import torch

# Automatically select GPU if available, else CPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Choose a stronger model if you have the resources
# Options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO("yolov8l.pt")  # Use yolov8l.pt for even better accuracy if you have enough VRAM

# Training configuration
model.train(
    data="planner.v2i.yolov8/data.yaml",
    epochs=300,
    imgsz=768,
    batch=8,
    lr0=0.001,
    lrf=0.01,
    patience=50,
    optimizer='auto',
    augment=True,
    mosaic=1.0,  # or mosaic=0.8
    mixup=0.2,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.2,
    translate=0.1,
    scale=0.5,
    shear=0.1,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.5,
    workers=4,
    device=device,
    project="./train_model",
    name="yolov8l_custom"
)

# After training, evaluate on the test set
metrics = model.val(data="planner.v2i.yolov8/data.yaml", split='test')
print(metrics)
