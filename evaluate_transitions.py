import os
import cv2
from ultralytics import YOLO

from world_model import WorldModel


INPUT_DIR = "datasets/restricted_zone"

ZONE_X1 = 400
ZONE_Y1 = 100
ZONE_X2 = 600
ZONE_Y2 = 400


EXPECTED_SEQUENCES = {
    "outside": [
        "OUTSIDE"
    ],
    "entering": [
        "OUTSIDE",
        "ENTERING",
        "STAYING_INSIDE"
    ],
    "staying_inside": [
        "STAYING_INSIDE"
    ],
    "leaving": [
        "STAYING_INSIDE",
        "LEAVING",
        "OUTSIDE"
    ]
}


model = YOLO("yolov8n.pt")


def get_state_sequence(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return []

    world_model = WorldModel()

    states = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        frame_states = []

        if result.boxes is not None:

            boxes = result.boxes

            ids = boxes.id
            classes = boxes.cls
            xyxy = boxes.xyxy

            if ids is not None:

                ids = ids.cpu().numpy()
                classes = classes.cpu().numpy()
                xyxy = xyxy.cpu().numpy()

                for i in range(len(ids)):

                    object_id = int(ids[i])
                    class_id = int(classes[i])

                    if class_id != 0:
                        continue

                    x1, y1, x2, y2 = xyxy[i]

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

                    frame_states.append(
                        state["zone_state"]
                    )

        if frame_states:

            if "ENTERING" in frame_states:
                current_state = "ENTERING"

            elif "LEAVING" in frame_states:
                current_state = "LEAVING"

            elif "STAYING_INSIDE" in frame_states:
                current_state = "STAYING_INSIDE"

            elif "OUTSIDE" in frame_states:
                current_state = "OUTSIDE"

            else:
                current_state = frame_states[0]

            if not states or states[-1] != current_state:
                states.append(current_state)

    cap.release()

    return states


def clean_sequence(states):

    cleaned = []

    for state in states:

        if state not in cleaned:
            cleaned.append(state)

    return cleaned


def check_sequence(actual, expected):

    actual = clean_sequence(actual)

    if expected == ["OUTSIDE"]:

        return "OUTSIDE" in actual

    if expected == ["STAYING_INSIDE"]:

        return "STAYING_INSIDE" in actual

    if expected == [
        "OUTSIDE",
        "ENTERING",
        "STAYING_INSIDE"
    ]:

        return (
            "OUTSIDE" in actual
            and
            "ENTERING" in actual
            and
            "STAYING_INSIDE" in actual
        )

    if expected == [
        "STAYING_INSIDE",
        "LEAVING",
        "OUTSIDE"
    ]:

        return (
            "STAYING_INSIDE" in actual
            and
            "LEAVING" in actual
            and
            "OUTSIDE" in actual
        )

    return False


def main():

    total_videos = 0
    successful_videos = 0

    category_results = {
        "outside": [0, 0],
        "entering": [0, 0],
        "staying_inside": [0, 0],
        "leaving": [0, 0]
    }

    print("\n==========================================")
    print("WORLD MODEL TRANSITION EVALUATION")
    print("==========================================")

    for category, expected in EXPECTED_SEQUENCES.items():

        folder = os.path.join(
            INPUT_DIR,
            category
        )

        if not os.path.exists(folder):
            continue

        videos = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(".mp4")
        ])

        print(f"\nCATEGORY: {category.upper()}")

        for video in videos:

            video_path = os.path.join(
                folder,
                video
            )

            print(
                f"\nEvaluating: {video}"
            )

            actual = get_state_sequence(
                video_path
            )

            print(
                "Detected sequence:"
            )

            print(
                " -> ".join(actual)
            )

            success = check_sequence(
                actual,
                expected
            )

            total_videos += 1

            category_results[category][1] += 1

            if success:

                successful_videos += 1

                category_results[category][0] += 1

                print("Result: PASS")

            else:

                print("Result: FAIL")

            print(
                "Expected:"
            )

            print(
                " -> ".join(expected)
            )

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")

    for category in EXPECTED_SEQUENCES:

        passed = category_results[category][0]

        total = category_results[category][1]

        accuracy = (
            passed / total * 100
            if total > 0
            else 0
        )

        print(
            f"{category.upper():20} "
            f"{passed}/{total} "
            f"({accuracy:.2f}%)"
        )

    overall_accuracy = (
        successful_videos / total_videos * 100
        if total_videos > 0
        else 0
    )

    print("------------------------------------------")

    print(
        f"OVERALL: "
        f"{successful_videos}/{total_videos} "
        f"({overall_accuracy:.2f}%)"
    )

    print("==========================================")


if __name__ == "__main__":
    main()