import os
import cv2
import time

from ultralytics import YOLO

from world_model import WorldModel
from decision_engine import make_decision
from vlm_reasoner import analyze_scene
from event_logger import log_event


# ============================================================
# CONFIGURATION
# ============================================================

ZONE_X1 = 400
ZONE_Y1 = 100
ZONE_X2 = 600
ZONE_Y2 = 400

VLM_INTERVAL = 3

RECORDING_FPS = 20.0

RECORDING_ROOT = "datasets/restricted_zone"

CATEGORIES = {
    ord("1"): "outside",
    ord("2"): "entering",
    ord("3"): "staying_inside",
    ord("4"): "leaving"
}


# ============================================================
# CREATE RECORDING FOLDERS
# ============================================================

for category in CATEGORIES.values():
    os.makedirs(
        os.path.join(RECORDING_ROOT, category),
        exist_ok=True
    )


def get_next_recording_path(category):

    folder = os.path.join(
        RECORDING_ROOT,
        category
    )

    os.makedirs(folder, exist_ok=True)

    number = 1

    while True:

        filename = f"{category}_{number:02d}.mp4"

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.exists(path):
            return path

        number += 1


# ============================================================
# INITIALIZATION
# ============================================================

print("\nLoading YOLO model...")

model = YOLO("yolov8s.pt")

print("YOLO loaded successfully.")

world_model = WorldModel()

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    exit()

print("Camera started.")


# ============================================================
# VLM VARIABLES
# ============================================================

vlm_description = ""

last_vlm_time = 0

vlm_running = False


# ============================================================
# EVENT STATE
# ============================================================

last_risk_state = "LOW"


# ============================================================
# TRACKING PERFORMANCE
# ============================================================

previous_person_ids = set()

id_switches = 0

frames_without_id_switch = 0

total_frames = 0

total_processing_time = 0


# ============================================================
# RECORDING VARIABLES
# ============================================================

recording = False

video_writer = None

recording_category = "outside"

recording_path = ""

recording_frame_count = 0


# ============================================================
# DISPLAY INSTRUCTIONS
# ============================================================

print("\nControls:")

print("1 = OUTSIDE")

print("2 = ENTERING")

print("3 = STAYING_INSIDE")

print("4 = LEAVING")

print("R = Start / Stop Recording")

print("V = Analyze Scene using VLM")

print("Q = Quit")

print("\nCurrent recording category: OUTSIDE")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    loop_start = time.time()

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")

        break

    total_frames += 1

    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    # ========================================================
    # DRAW RESTRICTED ZONE
    # ========================================================

    cv2.rectangle(
        annotated_frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        annotated_frame,
        "RESTRICTED ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # ========================================================
    # TRACKING VARIABLES
    # ========================================================

    current_person_ids = set()

    person_inside_now = False

    person_detected_this_frame = False

    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

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

                x1, y1, x2, y2 = xyxy[i]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # =================================================
                # PERSON DETECTION
                # COCO CLASS 0 = PERSON
                # =================================================

                if class_id == 0:

                    person_detected_this_frame = True

                    current_person_ids.add(object_id)

                    # =============================================
                    # CHECK RESTRICTED ZONE
                    # =============================================

                    inside_zone = (
                        ZONE_X1 <= center_x <= ZONE_X2
                        and
                        ZONE_Y1 <= center_y <= ZONE_Y2
                    )

                    if inside_zone:

                        person_inside_now = True

                    # =============================================
                    # WORLD MODEL UPDATE
                    # =============================================

                    world_model.update(
                        object_id,
                        "person",
                        (center_x, center_y),
                        inside_zone
                    )

                    # =============================================
                    # TRACKING INFORMATION
                    # =============================================

                    cv2.circle(
                        annotated_frame,
                        (center_x, center_y),
                        5,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        annotated_frame,
                        f"ID: {object_id}",
                        (center_x - 30, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

                    # =============================================
                    # MOVEMENT / ZONE STATE
                    # =============================================

                    current_state = world_model.get_current_state()

                    if current_state is not None:

                        state_text = str(current_state)

                        cv2.putText(
                            annotated_frame,
                            state_text,
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

    # ========================================================
    # TRACKING ID PERFORMANCE
    # ========================================================

    if len(current_person_ids) > 0:

        if len(previous_person_ids) > 0:

            if current_person_ids != previous_person_ids:

                id_switches += 1

            else:

                frames_without_id_switch += 1

        else:

            frames_without_id_switch += 1

    previous_person_ids = current_person_ids.copy()

    # ========================================================
    # DECISION ENGINE
    # ========================================================

    try:
        current_state = world_model.get_current_state()

        # Keep only objects detected in the current frame
        active_state = {
            obj_id: obj
            for obj_id, obj in current_state.items()
        if obj_id in current_person_ids
        }

        decision = make_decision(
            active_state,
            vlm_description
        )

        risk = decision["risk"]
        reason = decision["reason"]
        decision_text = f"{risk}: {reason}"

    except Exception as e:
        risk = "UNKNOWN"
        reason = "Decision engine error"
        decision_text = str(e)
        print("Decision Engine Error:", e)

    # ========================================================
    # EVENT LOGGING
    # ========================================================

    if person_inside_now:

        if last_risk_state != "HIGH":

            event_reason = "Person detected inside restricted zone"

            if vlm_description:

                event_reason += (
                    " | VLM: "
                    + vlm_description
                )

            log_event(
                "RESTRICTED_ZONE_INTRUSION",
                "HIGH",
                event_reason
            )

        last_risk_state = "HIGH"

    else:

        last_risk_state = "LOW"

    # ========================================================
    # DISPLAY RISK
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Risk: {risk}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if risk == "HIGH" else (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Reason: {reason}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY DECISION
    # ========================================================

    if decision_text:

        cv2.putText(
            annotated_frame,
            f"Decision: {decision_text}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ========================================================
    # DISPLAY VLM RESULT
    # ========================================================

    if vlm_description:

        cv2.putText(
            annotated_frame,
            "VLM:",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            vlm_description[:90],
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    # ========================================================
    # PROCESSING TIME + FPS
    # ========================================================

    processing_time = time.time() - loop_start

    total_processing_time += processing_time

    fps = 1 / processing_time if processing_time > 0 else 0

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Processing: {processing_time * 1000:.2f} ms",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ========================================================
    # RECORDING STATUS
    # ========================================================

    if recording:

        cv2.putText(
            annotated_frame,
            "● REC",
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Category: {recording_category}",
            (20, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        if video_writer is not None:

            video_writer.write(
                annotated_frame
            )

            recording_frame_count += 1

    else:

        cv2.putText(
            annotated_frame,
            f"Category: {recording_category}",
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Hybrid World Model Intelligence Framework",
        annotated_frame
    )

    # ========================================================
    # KEY INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # CATEGORY SELECTION
    # ========================================================

    if key in CATEGORIES:

        if recording:

            print(
                "\nStop recording before changing category."
            )

        else:

            recording_category = CATEGORIES[key]

            print(
                f"\nRecording category changed to: "
                f"{recording_category.upper()}"
            )

    # ========================================================
    # R = START / STOP RECORDING
    # ========================================================

    elif key == ord("r") or key == ord("R"):

        if not recording:

            recording_path = get_next_recording_path(
                recording_category
            )

            height, width = annotated_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            video_writer = cv2.VideoWriter(
                recording_path,
                fourcc,
                RECORDING_FPS,
                (width, height)
            )

            if not video_writer.isOpened():

                print(
                    "\nERROR: Could not start video recording."
                )

                video_writer = None

            else:

                recording = True

                recording_frame_count = 0

                print("\n==============================")

                print("RECORDING STARTED")

                print(
                    f"Category: "
                    f"{recording_category}"
                )

                print(
                    f"File: "
                    f"{recording_path}"
                )

                print("==============================\n")

        else:

            recording = False

            if video_writer is not None:

                video_writer.release()

                video_writer = None

            print("\n==============================")

            print("RECORDING STOPPED")

            print(
                f"Saved: {recording_path}"
            )

            print(
                f"Frames recorded: "
                f"{recording_frame_count}"
            )

            print("==============================\n")

    # ========================================================
    # V = VLM ANALYSIS
    # ========================================================

    elif key == ord("v") or key == ord("V"):

        if not vlm_running:

            vlm_running = True

            print("\nRunning VLM analysis...")

            try:

                vlm_description = analyze_scene(
                    frame
                )

                print(
                    "VLM:",
                    vlm_description
                )

                last_vlm_time = time.time()

            except Exception as e:

                print(
                    "VLM Error:",
                    e
                )

            vlm_running = False

    # ========================================================
    # Q = QUIT
    # ========================================================

    elif key == ord("q") or key == ord("Q"):

        break


# ============================================================
# STOP RECORDING IF ACTIVE
# ============================================================

if video_writer is not None:

    video_writer.release()

    video_writer = None


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# PERFORMANCE REPORT
# ============================================================

print("\n")
print("==========================================")
print("     HYBRID WORLD MODEL PERFORMANCE")
print("==========================================")

print(
    f"Total Frames: {total_frames}"
)

if total_frames > 0:

    average_processing_time = (
        total_processing_time
        / total_frames
    )

    average_fps = (
        total_frames
        / total_processing_time
        if total_processing_time > 0
        else 0
    )

    print(
        f"Average FPS: "
        f"{average_fps:.2f}"
    )

    print(
        f"Average Processing Time: "
        f"{average_processing_time * 1000:.2f} ms/frame"
    )

else:

    print(
        "Average FPS: 0"
    )

    print(
        "Average Processing Time: 0 ms/frame"
    )


print(
    f"Frames Without ID Switch: "
    f"{frames_without_id_switch}"
)

print(
    f"ID Switches: "
    f"{id_switches}"
)


if total_frames > 0:

    id_stability = (
        frames_without_id_switch
        / total_frames
    ) * 100

else:

    id_stability = 0


print(
    f"ID Stability: "
    f"{id_stability:.2f}%"
)


if total_frames > 0:

    print(
        "Tracking Successful: YES"
    )

else:

    print(
        "Tracking Successful: NO"
    )


print("==========================================")
print("Program ended.")
print("==========================================")