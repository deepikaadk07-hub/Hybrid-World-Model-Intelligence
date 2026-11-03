import json
from pathlib import Path

BASE = Path(r"C:\Users\deepi\hybrid_world_model\datasets\COCO")

SOURCE_IMAGES = BASE / "images" / "val2017"
SOURCE_JSON = BASE / "annotations" / "instances_val2017.json"
OUTPUT_JSON = BASE / "annotations" / "instances_val2017_subset.json"

if not SOURCE_IMAGES.is_dir():
    raise FileNotFoundError(f"Images folder not found: {SOURCE_IMAGES}")

if not SOURCE_JSON.is_file():
    raise FileNotFoundError(f"Original annotation file not found: {SOURCE_JSON}")

with SOURCE_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_files = {
    p.name
    for p in SOURCE_IMAGES.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

selected_images = [
    image for image in data["images"]
    if image["file_name"] in image_files
]

selected_ids = {image["id"] for image in selected_images}

subset = {
    **data,
    "images": selected_images,
    "annotations": [
        ann for ann in data["annotations"]
        if ann["image_id"] in selected_ids
    ]
}

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(subset, f)

print("COCO annotations filtered successfully!")
print(f"Images matched: {len(selected_images)}")
print(f"Annotations retained: {len(subset['annotations'])}")
print(f"Output file: {OUTPUT_JSON}")