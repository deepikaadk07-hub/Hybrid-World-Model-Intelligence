import json
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\Users\deepi\hybrid_world_model\datasets\COCO")

with (BASE / "annotations" / "instances_val2017_subset.json").open(
    "r", encoding="utf-8"
) as f:
    data = json.load(f)

with (BASE / "predictions.json").open("r", encoding="utf-8") as f:
    predictions = json.load(f)

print("Images in annotations:", len(data["images"]))
print("Ground-truth annotations:", len(data["annotations"]))
print("Predictions:", len(predictions))

print("\nGround-truth category IDs:")
print(sorted({a["category_id"] for a in data["annotations"]}))

print("\nPredicted category IDs:")
print(sorted({p["category_id"] for p in predictions}))

print("\nPredictions per image:")
counts = Counter(p["image_id"] for p in predictions)
print("Maximum:", max(counts.values(), default=0))
print("Average:", round(len(predictions) / max(len(data["images"]), 1), 2))