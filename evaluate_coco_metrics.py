from pathlib import Path
import json
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = Path(r"C:\Users\deepi\hybrid_world_model\datasets\COCO")
IMAGES = BASE / "images" / "val2017"
ANNOTATIONS = BASE / "annotations" / "instances_val2017_subset.json"
PREDICTIONS = BASE / "predictions.json"

COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def main():
    if not IMAGES.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGES}")

    if not ANNOTATIONS.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATIONS}")

    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    image_ids = {image["id"] for image in annotation_data["images"]}
    image_files = [
        IMAGES / image["file_name"]
        for image in annotation_data["images"]
        if (IMAGES / image["file_name"]).exists()
    ]

    print("Images listed in annotations:", len(image_ids))
    print("Image files found:", len(image_files))
    print("Ground-truth annotations:", len(annotation_data["annotations"]))

    if not image_files:
        raise RuntimeError("No matching images found. Check image filenames.")

    model = YOLO("yolov8n.pt")
    predictions = []

    results = model.predict(
        source=[str(path) for path in image_files],
        conf=0.001,
        iou=0.7,
        max_det=100,
        verbose=False
    )

    print("Images evaluated by YOLO:", len(results))

    max_detections = max((len(result.boxes) for result in results), default=0)
    print("Maximum detections in one image:", max_detections)

    for result in results:
        image_name = Path(result.path).name

        image_info = next(
            (
                image for image in annotation_data["images"]
                if image["file_name"] == image_name
            ),
            None
        )

        if image_info is None:
            continue

        image_id = image_info["id"]

        for box in result.boxes:
            class_id = int(box.cls.item())

            if class_id < 0 or class_id >= len(COCO_CATEGORY_IDS):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            if width <= 0 or height <= 0:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": COCO_CATEGORY_IDS[class_id],
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(box.conf.item())
            })

    with open(PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print("Total predictions:", len(predictions))
    print("Predictions saved to:", PREDICTIONS)

    if not predictions:
        raise RuntimeError("No predictions generated.")

    coco_gt = COCO(str(ANNOTATIONS))
    coco_dt = coco_gt.loadRes(str(PREDICTIONS))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.maxDets = [1, 10, 100]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    print("\nFinal COCO metrics:")
    print(f"mAP@0.50:0.95: {evaluator.stats[0]:.4f}")
    print(f"mAP@0.50:      {evaluator.stats[1]:.4f}")
    print(f"AP@0.75:       {evaluator.stats[2]:.4f}")
    print(f"AR@100:        {evaluator.stats[8]:.4f}")


if __name__ == "__main__":
    main()