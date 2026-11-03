
import os
import csv
import cv2
from ultralytics import YOLO
from world_model import WorldModel

DATA_DIR = "datasets/restricted_zone"
OUTPUT_FILE = "world_model_test_data.csv"

ZONE_X1, ZONE_Y1 = 400, 100
ZONE_X2, ZONE_Y2 = 600, 400

model = YOLO("yolov8n.pt")

# Write a fresh, consistent CSV header.
# Keep the existing file backed up if it contains data.
if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    print(f"{OUTPUT_FILE} already contains data.")
    print("Rename or back it up before collecting new predictions.")
    raise SystemExit

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "video", "frame", "object_id",
        "actual_state", "predicted_state"
    ])

    for category in ["entering"]:
        folder = os.path.join(DATA_DIR, category)

        for filename in sorted(os.listdir(folder))[:1]:
            if not filename.lower().endswith(
                (".mp4", ".avi", ".mov")
            ):
                continue

            video_path = os.path.join(folder, filename)
            print("Processing:", video_path)

            # Reset history for every video.
            world_model = WorldModel()
            world_model.csv_file = "temporary_collection_log.csv"

            # Remove stale IDs between videos.
            world_model.objects.clear()
            world_model.history.clear()

            cap = cv2.VideoCapture(video_path)
            frame_number = 0

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                for result in results:
                    if result.boxes is None:
                        continue
                    if result.boxes.id is None:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()

                    for box, object_id, class_id in zip(
                        boxes, ids, classes
                    ):
                        if model.names[class_id] != "person":
                            continue

                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        inside = (
                            ZONE_X1 <= cx <= ZONE_X2
                            and ZONE_Y1 <= cy <= ZONE_Y2
                        )

                        state = world_model.update(
                            object_id,
                            "person",
                            (cx, cy),
                            inside
                        )

                        writer.writerow([
                            filename,
                            frame_number,
                            object_id,
                            "",
                            state["zone_state"]
                        ])

                if frame_number % 100 == 0:
                    print("  Processed frames:", frame_number)

            cap.release()

print("Prediction collection complete.")
print("Ground-truth labels remain blank for manual verification.")
