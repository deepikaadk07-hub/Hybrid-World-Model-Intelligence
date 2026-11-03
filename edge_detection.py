from ultralytics import YOLO
import cv2

print("Starting program...")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("Camera opened successfully")

while True:

    # Read camera frame
    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read camera frame")
        break

    # STEP 5: Object detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # STEP 6: Extract object information
    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, object_id, class_id in zip(
            boxes,
            ids,
            classes
        ):

            x1, y1, x2, y2 = box

            object_name = model.names[class_id]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            print(
                "ID:", object_id,
                "| Object:", object_name,
                "| Position:", center_x, center_y
            )

    # Draw YOLO tracking boxes
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLO Tracking", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Program closed successfully")