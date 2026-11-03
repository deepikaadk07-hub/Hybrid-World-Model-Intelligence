import cv2
import time
from vlm_reasoner import analyze_scene

video_path = "datasets/restricted_zone/entering/entering_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    raise SystemExit

frame_numbers = [10, 20, 30, 40, 50, 60]
times = []

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_number}")
        continue

    start = time.perf_counter()
    description = analyze_scene(frame)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(f"\nFrame: {frame_number}")
    print(f"Caption: {description}")
    print(f"Time: {elapsed:.3f} seconds")

cap.release()

if times:
    print(f"\nAverage inference time: {sum(times) / len(times):.3f} seconds")