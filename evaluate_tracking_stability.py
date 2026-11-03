
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

VIDEO_ROOT = Path("datasets/restricted_zone")
OUTPUT_FILE = Path("tracking_stability_results.csv")
MODEL_PATH = "yolov8n.pt"

CATEGORIES = ["entering", "leaving"]
model = YOLO(MODEL_PATH)
results = []

for category in CATEGORIES:
    folder = VIDEO_ROOT / category

    for video_path in sorted(folder.glob("*.mp4")):
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(str(video_path))

        total_frames = 0
        frames_with_person = 0
        frames_with_id = 0
        observed_ids = set()
        previous_ids = set()
        id_change_events = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            total_frames += 1

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.15,
                verbose=False
            )[0]

            current_ids = set()

            if result.boxes is not None and len(result.boxes) > 0:
                frames_with_person += 1

                if result.boxes.id is not None:
                    current_ids = {
                        int(track_id)
                        for track_id in result.boxes.id.cpu().tolist()
                    }

            if current_ids:
                frames_with_id += 1
                observed_ids.update(current_ids)

                if previous_ids and current_ids.isdisjoint(previous_ids):
                    id_change_events += 1

            previous_ids = current_ids
           
        cap.release()

        track_coverage = (
            frames_with_id / total_frames * 100
            if total_frames else 0
        )

        results.append({
            "Category": category,
            "Video": video_path.name,
            "Total_Frames": total_frames,
            "Frames_With_Person": frames_with_person,
            "Frames_With_Track_ID": frames_with_id,
            "Track_Coverage_Percent": round(track_coverage, 2),
            "Unique_Track_IDs": len(observed_ids),
            "ID_Change_Events_Proxy": id_change_events
        })

        print(
            f"{video_path.name}: "
            f"coverage={track_coverage:.2f}%, "
            f"unique IDs={len(observed_ids)}, "
            f"ID change events={id_change_events}"
        )

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")
