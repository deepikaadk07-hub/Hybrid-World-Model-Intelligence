
import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO
from world_model import WorldModel
from decision_engine import make_decision

VIDEO_ROOT = Path("datasets/restricted_zone")
OUTPUT_FILE = Path("proposed_results.csv")
MODEL_PATH = "yolov8s.pt"

ZONE_X1, ZONE_Y1 = 400, 100
ZONE_X2, ZONE_Y2 = 600, 400

CATEGORIES = [
    "entering",
    "leaving",
    "outside",
    "staying_inside"
]

results = []

for category in CATEGORIES:
    for video_path in sorted((VIDEO_ROOT / category).glob("*.mp4")):
        model = YOLO(MODEL_PATH)
        world_model = WorldModel()

        cap = cv2.VideoCapture(str(video_path))

        total_frames = 0
        detected_frames = 0
        tracked_frames = 0
        inside_frames = 0
        high_risk_frames = 0
        total_time = 0.0
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

            current_inside = False

            if result.boxes is not None and len(result.boxes) > 0:
                detected_frames += 1
                boxes = result.boxes

                if boxes.id is not None:
                    tracked_frames += 1
                    ids = boxes.id.cpu().tolist()
                    coordinates = boxes.xyxy.cpu().tolist()

                    for track_id, box in zip(ids, coordinates):
                        track_id = int(track_id)
                        observed_ids.add(track_id)

                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        inside = (
                            ZONE_X1 <= cx <= ZONE_X2
                            and ZONE_Y1 <= cy <= ZONE_Y2
                        )

                        current_inside |= inside

                        world_model.update(
                            object_id=track_id,
                            object_class="person",
                            position=(cx, cy),
                            inside_zone=inside
                        )

            if current_inside:
                inside_frames += 1

            decision = make_decision(
                world_model.get_current_state()
            )

            if decision["risk"] == "HIGH":
                high_risk_frames += 1

            total_time += time.perf_counter() - start

        cap.release()

        row = {
            "video": video_path.name,
            "category": category.upper(),
            "frames": total_frames,
            "detection_frame_percent": round(
                100 * detected_frames / total_frames, 2
            ) if total_frames else 0,
            "tracked_frames": tracked_frames,
            "inside_zone_frame_percent": round(
                100 * inside_frames / total_frames, 2
            ) if total_frames else 0,
            "high_risk_frame_percent": round(
                100 * high_risk_frames / total_frames, 2
            ) if total_frames else 0,
            "unique_track_ids": len(observed_ids),
            "avg_processing_ms": round(
                1000 * total_time / total_frames, 2
            ) if total_frames else 0,
            "fps": round(
                total_frames / total_time, 2
            ) if total_time else 0
        }

        results.append(row)

        print(
            f"{video_path.name}: "
            f"FPS={row['fps']:.2f}, "
            f"risk frames={high_risk_frames}, "
            f"inside-zone frames={inside_frames}"
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
