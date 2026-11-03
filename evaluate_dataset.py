
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("datasets/restricted_zone")
OUTPUT = Path("evaluation_results.csv")

ZONE_X1, ZONE_Y1 = 400, 100
ZONE_X2, ZONE_Y2 = 600, 400

STATES = ["outside", "entering", "staying_inside", "leaving"]

model = YOLO("yolov8s.pt")
rows = []

for state in STATES:
    folder = ROOT / state

    for video_path in sorted(folder.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path}")
            continue

        frame_no = 0
        person_frames = 0
        inside_frames = 0
        inference_times = []

        print(f"Evaluating: {video_path.name}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_no += 1
            start = time.perf_counter()

            result = model(frame, verbose=False)[0]

            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            detected_person = False
            inside_person = False

            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls.item()) != 0:
                        continue

                    detected_person = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if (ZONE_X1 <= cx <= ZONE_X2
                            and ZONE_Y1 <= cy <= ZONE_Y2):
                        inside_person = True

            person_frames += int(detected_person)
            inside_frames += int(inside_person)

        cap.release()

        avg_ms = (
            sum(inference_times) / len(inference_times) * 1000
            if inference_times else 0
        )

        rows.append({
            "video": video_path.name,
            "folder_label": state.upper(),
            "frames": frame_no,
            "frames_with_person": person_frames,
            "frames_with_person_inside_zone": inside_frames,
            "person_frame_percentage": round(
                100 * person_frames / frame_no, 2
            ) if frame_no else 0,
            "inside_zone_frame_percentage": round(
                100 * inside_frames / frame_no, 2
            ) if frame_no else 0,
            "mean_yolo_inference_ms": round(avg_ms, 2)
        })

        print(
            f"  Frames: {frame_no}, "
            f"person frames: {person_frames}, "
            f"inside-zone frames: {inside_frames}, "
            f"mean YOLO inference: {avg_ms:.2f} ms"
        )

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nFinished. Evaluated {len(rows)} videos.")
print(f"Results saved to: {OUTPUT}")
