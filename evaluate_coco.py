import json
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"C:\Users\deepi\hybrid_world_model\datasets\COCO")
IMAGES = BASE / "images" / "val2017"
ANNOTATIONS = BASE / "annotations" / "instances_val2017_subset.json"

MODEL = "yolov8n.pt"

with ANNOTATIONS.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_files = [
    IMAGES / image["file_name"]
    for image in data["images"]
    if (IMAGES / image["file_name"]).exists()
]

print("Images available:", len(image_files))

model = YOLO(MODEL)

total_detections = 0

for image_path in image_files:
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )
    total_detections += len(results[0].boxes)

print("\nCOCO evaluation completed!")
print("Images evaluated:", len(image_files))
print("Total detections:", total_detections)
print("Average detections per image:", round(
    total_detections / max(len(image_files), 1), 2
))