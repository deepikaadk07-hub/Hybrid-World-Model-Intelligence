import cv2
import csv
from pathlib import Path

VIDEO_ROOT = Path("datasets/restricted_zone")
OUTPUT_FILE = Path("ground_truth_states.csv")

VIDEOS = [
    "entering_01.mp4",
    "leaving_04.mp4",
    "outside_02.mp4",
    "staying_inside_01.mp4",
]

STATES = {
    ord("1"): "OUTSIDE",
    ord("2"): "ENTERING",
    ord("3"): "STAYING_INSIDE",
    ord("4"): "LEAVING",
}

rows = []

for video_name in VIDEOS:
    video_path = next(
        (VIDEO_ROOT / category / video_name
         for category in ["entering", "leaving", "outside", "staying_inside"]
         if (VIDEO_ROOT / category / video_name).exists()),
        None
    )

    if video_path is None:
        print(f"Video not found: {video_name}")
        continue

    cap = cv2.VideoCapture(str(video_path))
    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_number % 10 == 0:
            display = frame.copy()

            cv2.putText(
                display,
                f"{video_name} | Frame {frame_number}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.putText(
                display,
                "1:OUTSIDE 2:ENTERING 3:STAYING 4:LEAVING 5:SKIP Q:QUIT",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )

            cv2.imshow("Ground Truth Annotation", display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Annotation stopped.")
                raise SystemExit

            if key == ord("5"):
                frame_number += 1
                continue

            state = STATES.get(key)

            if state is None:
                print("Invalid key. Frame skipped.")
                frame_number += 1
                continue

            rows.append({
                "video": video_name,
                "frame_number": frame_number,
                "person_label": "Person_1",
                "ground_truth_state": state
            })

        frame_number += 1

    cap.release()

cv2.destroyAllWindows()

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "video", "frame_number",
        "person_label", "ground_truth_state"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved annotations to: {OUTPUT_FILE.resolve()}")