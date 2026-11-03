
import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO

VIDEO_ROOT = Path("datasets/restricted_zone")
OUTPUT_FILE = Path("baseline_results.csv")
MODEL_PATH = "yolov8s.pt"

ZONE_X1, ZONE_Y1 = 400, 100
ZONE_X2, ZONE_Y2 = 600, 400

CATEGORIES = [
    "entering",
    "leaving",
    "outside",
    "staying_inside"
]

model = YOLO(MODEL_PATH)
results = []

for category in CATEGORIES:
    folder = VIDEO_ROOT / category

    for video_path in sorted(folder.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))

        total_frames = 0
        detected_frames = 0
        inside_frames = 0
        tracked_frames = 0
        total_processing_time = 0.0
        observed_ids = set()

        while True:
            success, frame = cap.read()
            if not success:
                break

            total_frames += 1
            start = time.perf_counter()

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False
            )[0]

            total_processing_time += time.perf_counter() - start

            if result.boxes is not None and len(result.boxes) > 0:
                detected_frames += 1

                boxes = result.boxes
                if boxes.id is not None:
                    tracked_frames += 1
                    ids = boxes.id.cpu().tolist()
                    observed_ids.update(int(i) for i in ids)

                    for box in boxes.xyxy.cpu().tolist():
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        if (
                            ZONE_X1 <= cx <= ZONE_X2
                            and ZONE_Y1 <= cy <= ZONE_Y2
                        ):
                            inside_frames += 1
                            break

        cap.release()

        fps = (
            total_frames / total_processing_time
            if total_processing_time > 0 else 0
        )

        row = {
            "video": video_path.name,
            "category": category.upper(),
            "frames": total_frames,
            "person_detection_frames": detected_frames,
            "tracked_frames": tracked_frames,
            "inside_zone_frames": inside_frames,
            "detection_frame_percent": round(
                100 * detected_frames / total_frames, 2
            ) if total_frames else 0,
            "inside_zone_frame_percent": round(
                100 * inside_frames / total_frames, 2
            ) if total_frames else 0,
            "unique_track_ids": len(observed_ids),
            "avg_processing_ms": round(
                1000 * total_processing_time / total_frames, 2
            ) if total_frames else 0,
            "fps": round(fps, 2)
        }

        results.append(row)
        print(
            f"{video_path.name}: "
            f"FPS={fps:.2f}, "
            f"inside-zone frames={inside_frames}, "
            f"unique IDs={len(observed_ids)}"
        )

if results:
    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {OUTPUT_FILE.resolve()}")
else:
    print("No videos found. Check VIDEO_ROOT.")
