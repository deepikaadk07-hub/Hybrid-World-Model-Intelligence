
from pathlib import Path
import json
import time
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = Path(r"C:\Users\deepi\hybrid_world_model\datasets\COCO")
IMAGES = BASE / "images" / "val2017"
ANNOTATIONS = BASE / "annotations" / "instances_val2017_subset.json"

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

def evaluate_model(model_name, image_files, image_info_by_name, gt_image_ids):
    print(f"\nEvaluating {model_name}...")
    model = YOLO(model_name)
    predictions = []

    start = time.perf_counter()

    results = model.predict(
        source=[str(p) for p in image_files],
        conf=0.001,
        iou=0.7,
        max_det=100,
        verbose=False
    )

    elapsed = time.perf_counter() - start

    for result in results:
        name = Path(result.path).name
        image_info = image_info_by_name.get(name)

        if image_info is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())

            if not 0 <= class_id < len(COCO_CATEGORY_IDS):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            predictions.append({
                "image_id": image_info["id"],
                "category_id": COCO_CATEGORY_IDS[class_id],
                "bbox": [x1, y1, width, height],
                "score": float(box.conf.item())
            })

    prediction_file = BASE / f"predictions_{model_name.replace('.pt', '')}.json"
    with open(prediction_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    coco_gt = COCO(str(ANNOTATIONS))
    coco_dt = coco_gt.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(gt_image_ids)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "model": model_name,
        "images": len(results),
        "predictions": len(predictions),
        "mAP50_95": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR100": float(evaluator.stats[8]),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(results)
    }

def main():
    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_info_by_name = {
        image["file_name"]: image for image in data["images"]
    }

    image_files = [
        IMAGES / name for name in image_info_by_name
        if (IMAGES / name).exists()
    ]

    if len(image_files) != len(image_info_by_name):
        raise RuntimeError("Some annotated images are missing.")

    gt_image_ids = {image["id"] for image in data["images"]}

    results = []
    for model_name in ["yolov8n.pt", "yolov8s.pt"]:
        results.append(
            evaluate_model(
                model_name, image_files, image_info_by_name, gt_image_ids
            )
        )

    output_file = BASE / "model_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nMODEL COMPARISON")
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"mAP@0.50:0.95: {result['mAP50_95']:.4f}")
        print(f"mAP@0.50:      {result['mAP50']:.4f}")
        print(f"AP@0.75:       {result['AP75']:.4f}")
        print(f"AR@100:        {result['AR100']:.4f}")
        print(f"Seconds/image: {result['seconds_per_image']:.4f}")

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
