import os
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import numpy as np
import json

# Directory paths
IMAGE_DIR = "images"
OUTPUT_DIR = "output"
MODEL_PATH = "C:/pinokio/api/automatic1111.git/app/runs/detect/train3/weights/best.pt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load YOLOv8 Model (Trained model)
model = YOLO(MODEL_PATH)

# Load the default SAM model and move to device
sam = sam_model_registry['default']().to(DEVICE)
predictor = SamPredictor(sam)

def resize_image_keep_aspect(image, max_size=1024):
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def run_yolo_inference(image_path):
    print(f"Running YOLOv8 on {image_path}...")
    results = model.predict(source=image_path)
    print("YOLOv8 inference done.")
    return results

def pluralize(name):
    # Simple pluralization: add 's' unless it already ends with 's'
    return name if name.endswith('s') else name + 's'

def class_to_json_key(label):
    # Lowercase, replace spaces with underscores, pluralize
    return pluralize(label.lower().replace(' ', '_'))

def process_detection_and_segmentation(image_path, results):
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
    class_indices = results[0].boxes.cls.cpu().numpy().astype(int)  # [N]
    names = results[0].names if hasattr(results[0], 'names') else model.names  # class index to name

    # Read and resize image for SAM
    image = cv2.imread(image_path)
    image = resize_image_keep_aspect(image, max_size=1024)
    print(f"Image loaded and resized: {image.shape}")
    predictor.set_image(image, image_format="BGR")
    print("SAM image embedding computed.")

    # Collect all detected class names
    detected_classes = set()
    json_output = {}

    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        class_idx = class_indices[i]
        label = names[class_idx] if isinstance(names, dict) else names[class_idx]
        detected_classes.add(label)

        # Use bounding box as prompt for SAM (expects np.array shape (4,))
        box_prompt = np.array([x_min, y_min, x_max, y_max])
        print(f"Segmenting {label} at bbox: {box_prompt.tolist()}")
        masks, _, _ = predictor.predict(box=box_prompt[None, :], multimask_output=False)
        mask = masks[0]  # [H, W]
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.squeeze(1).tolist() for contour in contours if contour.shape[0] >= 3]

        # Determine the correct JSON key
        if label.lower() == "outerwall":
            key = "outer_walls"
        else:
            key = class_to_json_key(label)
        if key not in json_output:
            json_output[key] = []

        # Generic object structure
        obj = {
            "label": label,
            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "polygon": polygons
        }
        json_output[key].append(obj)

    print("\nDetected classes in this image:")
    for c in detected_classes:
        print(f"- {c}")
    print("Segmentation and JSON structuring done.")
    return json_output

def save_results_to_json(json_data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Results saved to {output_file}")

def main():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in the images directory.")
        return
    # Only process the first image
    image_name = image_files[0]
    image_path = os.path.join(IMAGE_DIR, image_name)
    print(f"\nProcessing {image_name}...")
    results = run_yolo_inference(image_path)
    json_data = process_detection_and_segmentation(image_path, results)
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}.json")
    save_results_to_json(json_data, output_path)
    print("\nSample JSON output for this image:")
    print(json.dumps(json_data, indent=2))

if __name__ == "__main__":
    main()
