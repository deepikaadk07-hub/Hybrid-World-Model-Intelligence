
import cv2
import csv
from pathlib import Path

ROOT = Path("datasets/restricted_zone")
CSV_FILE = Path("ground_truth_states.csv")

FOLDERS = ["entering", "leaving", "outside", "staying_inside"]

with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

states = {
    ord("1"): "OUTSIDE",
    ord("2"): "ENTERING",
    ord("3"): "STAYING_INSIDE",
    ord("4"): "LEAVING",
}

updated = []

for row in rows:
    video = row["video"]
    frame_no = int(row["frame_number"])

    path = next(
        (ROOT / folder / video for folder in FOLDERS
         if (ROOT / folder / video).exists()),
        None
    )

    if path is None:
        print("Missing video:", video)
        continue

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("Could not read:", video, frame_no)
        continue

    cv2.putText(
        frame, f"{video} | Frame {frame_no}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )
    cv2.putText(
        frame, f"Current label: {row['ground_truth_state']}",
        (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )
    cv2.putText(
        frame, "1 Outside | 2 Entering | 3 Staying | 4 Leaving",
        (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "5 Keep label | N No person/uncertain | Q Quit",
        (15, 120), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )

    cv2.imshow("Verify Ground Truth", frame)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord("q"):
        break
    elif key in states:
        row["ground_truth_state"] = states[key]
    elif key == ord("n"):
        row["ground_truth_state"] = "UNCERTAIN"
    # Key 5 or another key keeps the existing label.

    updated.append(row)

with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "video", "frame_number",
        "person_label", "ground_truth_state"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(updated)

print("Updated annotations:", CSV_FILE.resolve())
print("Rows reviewed:", len(updated))
