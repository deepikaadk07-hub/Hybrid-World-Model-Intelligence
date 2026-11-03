
import csv
from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from world_model import WorldModel

ROOT = Path("datasets/restricted_zone")
OUTPUT = Path("world_model_evaluation.csv")
TRANSITIONS_OUTPUT = Path("world_model_transitions.csv")

ZONE_X1, ZONE_Y1 = 400, 100
ZONE_X2, ZONE_Y2 = 600, 400

CATEGORIES = [
    "outside",
    "entering",
    "staying_inside",
    "leaving",
]

model = YOLO("yolov8s.pt")
video_rows = []
transition_rows = []

for category in CATEGORIES:
    folder = ROOT / category

    for video_path in sorted(folder.glob("*.mp4")):
        print(f"\nEvaluating: {video_path.name}")

        world = WorldModel()
        observed_states = Counter()
        transitions = []
        previous_state_by_id = {}
        total_frames = 0
        frames_with_person = 0

        # A separate tracking stream is created for each video.
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=False,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False,
        )

        for result in results:
            total_frames += 1
            frame_states = set()

            if result.boxes is None or result.boxes.id is None:
                continue

            ids = result.boxes.id.cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()

            if not ids:
                continue

            frames_with_person += 1

            for track_id, box in zip(ids, boxes):
                track_id = int(track_id)
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                inside_zone = (
                    ZONE_X1 <= cx <= ZONE_X2
                    and ZONE_Y1 <= cy <= ZONE_Y2
                )

                state = world.update(
                    track_id,
                    "person",
                    (cx, cy),
                    inside_zone,
                )

                current_state = state["zone_state"]
                observed_states[current_state] += 1
                frame_states.add(current_state)

                previous_state = previous_state_by_id.get(track_id)

                if (
                    previous_state is not None
                    and previous_state != current_state
                ):
                    transitions.append(
                        (track_id, previous_state, current_state)
                    )

                previous_state_by_id[track_id] = current_state

        state_sequence = " -> ".join(
            state for state in CATEGORIES
            if state.upper() in observed_states
        )

        video_rows.append({
            "video": video_path.name,
            "folder_label": category.upper(),
            "frames": total_frames,
            "frames_with_tracked_person": frames_with_person,
            "tracked_person_state_updates": sum(observed_states.values()),
            "outside_updates": observed_states["OUTSIDE"],
            "entering_updates": observed_states["ENTERING"],
            "staying_inside_updates": observed_states["STAYING_INSIDE"],
            "leaving_updates": observed_states["LEAVING"],
            "observed_states": "; ".join(
                state for state in [
                    "OUTSIDE",
                    "ENTERING",
                    "STAYING_INSIDE",
                    "LEAVING",
                ]
                if observed_states[state] > 0
            ),
            "observed_transition_count": len(transitions),
        })

        for track_id, from_state, to_state in transitions:
            transition_rows.append({
                "video": video_path.name,
                "folder_label": category.upper(),
                "track_id": track_id,
                "from_state": from_state,
                "to_state": to_state,
            })

        print("Observed states:", state_sequence or "None")
        print("Observed transitions:", len(transitions))

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=video_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(video_rows)

with TRANSITIONS_OUTPUT.open(
    "w", newline="", encoding="utf-8"
) as file:
    fields = [
        "video",
        "folder_label",
        "track_id",
        "from_state",
        "to_state",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(transition_rows)

print("\n========== EVALUATION SUMMARY ==========")
print("Videos evaluated:", len(video_rows))
print("Tracked state transitions:", len(transition_rows))
print("Saved:", OUTPUT)
print("Saved:", TRANSITIONS_OUTPUT)
