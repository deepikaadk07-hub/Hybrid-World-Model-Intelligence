import os
import cv2
import csv
from ultralytics import YOLO
from world_model import WorldModel

MODEL_PATH = "yolov8n.pt"
VIDEO_DIR = "datasets/restricted_zone"
OUTPUT_DIR = "transition_analysis"

ZONE_X1 = 400
ZONE_Y1 = 100
ZONE_X2 = 600
ZONE_Y2 = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

EXPECTED_SEQUENCES = {
    "entering": ["OUTSIDE", "ENTERING", "STAYING_INSIDE"],
    "leaving": ["STAYING_INSIDE", "LEAVING", "OUTSIDE"]
}

summary = []

for category in ["entering", "leaving"]:

    folder = os.path.join(VIDEO_DIR, category)

    for filename in sorted(os.listdir(folder)):

        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(folder, filename)

        csv_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(".mp4", "_analysis.csv")
        )

        cap = cv2.VideoCapture(video_path)
        world_model = WorldModel()

        rows = []
        frame_no = 0
        observed_states = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_no += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            if results[0].boxes is None:
                continue

            boxes = results[0].boxes

            for i in range(len(boxes)):

                if int(boxes.cls[i]) != 0:
                    continue

                if boxes.id is None:
                    continue

                object_id = int(boxes.id[i])

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                inside_zone = (
                    ZONE_X1 <= center_x <= ZONE_X2
                    and
                    ZONE_Y1 <= center_y <= ZONE_Y2
                )

                state = world_model.update(
                    object_id,
                    "person",
                    (center_x, center_y),
                    inside_zone
                )

                zone_state = state["zone_state"]

                observed_states.append(zone_state)

                rows.append([
                    frame_no,
                    object_id,
                    center_x,
                    center_y,
                    inside_zone,
                    zone_state,
                    state["movement"]
                ])

        cap.release()

        clean_states = []

        for state in observed_states:

            if not clean_states or clean_states[-1] != state:
                clean_states.append(state)

        expected = EXPECTED_SEQUENCES[category]

        index = 0

        for state in clean_states:

            if index < len(expected) and state == expected[index]:
                index += 1

        transition_detected = (
            "ENTERING" in clean_states
            if category == "entering"
            else "LEAVING" in clean_states
        )

        sequence_correct = index == len(expected)

        with open(csv_path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Frame",
                "Person_ID",
                "Center_X",
                "Center_Y",
                "Inside_Zone",
                "World_Model_State",
                "Movement"
            ])

            writer.writerows(rows)

        summary.append([
            category,
            filename,
            frame_no,
            " -> ".join(clean_states),
            "YES" if transition_detected else "NO",
            "YES" if sequence_correct else "NO"
        ])

summary_path = os.path.join(
    OUTPUT_DIR,
    "transition_summary.csv"
)

with open(summary_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Video",
        "Total_Frames",
        "Observed_State_Sequence",
        "Transition_Detected",
        "Expected_Sequence_Correct"
    ])

    writer.writerows(summary)

print("\n" + "=" * 90)
print("CSV-BASED TRANSITION ANALYSIS")
print("=" * 90)

for row in summary:

    print(
        f"{row[0].upper():12} | "
        f"{row[1]:20} | "
        f"Transition: {row[4]:3} | "
        f"Sequence: {row[5]:3} | "
        f"{row[3]}"
    )

print("\nCSV files saved in:")
print(OUTPUT_DIR)

print("\nSummary file:")
print(os.path.join(OUTPUT_DIR, "transition_summary.csv"))

print("=" * 90)