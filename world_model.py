
import time
import csv
import os


class WorldModel:

    def __init__(self, csv_file="state_predictions.csv"):
        self.objects = {}
        self.history = []
        self.csv_file = csv_file

        # Create a consistent CSV header for a new file
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "video",
                    "frame_number",
                    "object_id",
                    "actual_state",
                    "predicted_state"
                ])

    def update(
        self, object_id, object_class, position, inside_zone,
        video="", frame_number=-1, actual_state=""
    ):
        previous_position = None
        previous_zone_status = None

        if object_id in self.objects:
            previous_position = self.objects[object_id]["position"]
            previous_zone_status = self.objects[object_id]["inside_zone"]

        movement = "NEW"
        zone_state = "OUTSIDE"

        if previous_position is not None:
            old_x, old_y = previous_position
            new_x, new_y = position
            distance = (
                (new_x - old_x) ** 2 +
                (new_y - old_y) ** 2
            ) ** 0.5
            movement = "STILL" if distance < 5 else "MOVING"

        if previous_zone_status is None:
            zone_state = "STAYING_INSIDE" if inside_zone else "OUTSIDE"
        elif not previous_zone_status and inside_zone:
            zone_state = "ENTERING"
        elif previous_zone_status and inside_zone:
            zone_state = "STAYING_INSIDE"
        elif previous_zone_status and not inside_zone:
            zone_state = "LEAVING"
        else:
            zone_state = "OUTSIDE"

        state = {
            "id": object_id,
            "class": object_class,
            "position": position,
            "previous_position": previous_position,
            "movement": movement,
            "inside_zone": inside_zone,
            "zone_state": zone_state,
            "time": time.time(),
            "video": video,
            "frame_number": frame_number
        }

        self.objects[object_id] = state
        self.history.append(state)

        with open(self.csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                state["time"],
                video,
                frame_number,
                object_id,
                actual_state,
                zone_state
            ])

        return state

    def get_current_state(self):
        return self.objects

    def get_history(self):
        return self.history
