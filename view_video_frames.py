import cv2

video_path = "datasets/restricted_zone/entering/entering_01.mp4"

cap = cv2.VideoCapture(video_path)

# Start at frame 37 (frame index 36)
cap.set(cv2.CAP_PROP_POS_FRAMES, 36)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cv2.imshow(f"Frame {frame_number} - Press any key", frame)

    key = cv2.waitKey(0)

    if key == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()