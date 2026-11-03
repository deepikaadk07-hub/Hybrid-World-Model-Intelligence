import cv2
import os
from ultralytics import YOLO
from world_model import WorldModel

MODEL_PATH = "yolov8n.pt"

VIDEO_DIR = "datasets/restricted_zone"

ZONE_X1 = 400
ZONE_Y1 = 100
ZONE_X2 = 600
ZONE_Y2 = 400

model = YOLO(MODEL_PATH)

categories = ["entering", "leaving"]

for category in categories:

    folder = os.path.join(VIDEO_DIR, category)

    print("\n" + "=" * 70)
    print(category.upper())
    print("=" * 70)

    for filename in sorted(os.listdir(folder)):

        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(folder, filename)

        cap = cv2.VideoCapture(video_path)

        world_model = WorldModel()

        frame_no = 0
        previous_ids = set()
        transition_found = False

        print("\nVIDEO:", filename)

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

            current_ids = set()

            if results[0].boxes is not None:

                boxes = results[0].boxes

                for i in range(len(boxes)):

                    cls = int(boxes.cls[i])

                    if cls != 0:
                        continue

                    if boxes.id is None:
                        continue

                    object_id = int(boxes.id[i])
                    current_ids.add(object_id)

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

                    if zone_state in ["ENTERING", "LEAVING"]:

                        transition_found = True

                        print(
                            f"Frame {frame_no:4d} | "
                            f"ID {object_id:3d} | "
                            f"Position ({center_x:3d},{center_y:3d}) | "
                            f"Inside={inside_zone} | "
                            f"State={zone_state}"
                        )

            lost_ids = previous_ids - current_ids

            if lost_ids:
                print(
                    f"Frame {frame_no:4d} | "
                    f"WARNING: ID LOST -> {lost_ids}"
                )

            new_ids = current_ids - previous_ids

            if new_ids:
                print(
                    f"Frame {frame_no:4d} | "
                    f"NEW ID -> {new_ids}"
                )

            previous_ids = current_ids

        cap.release()

        if transition_found:
            print("RESULT: Transition detected")
        else:
            print("RESULT: NO TRANSITION DETECTED")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)