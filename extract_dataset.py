import os
import cv2

INPUT_DIR = "datasets/restricted_zone"
OUTPUT_DIR = "datasets/restricted_zone_frames"

CATEGORIES = [
    "outside",
    "entering",
    "staying_inside",
    "leaving"
]

FRAME_INTERVAL = 10


def extract_frames():

    total_frames = 0

    print("\nStarting frame extraction...\n")

    for category in CATEGORIES:

        input_folder = os.path.join(
            INPUT_DIR,
            category
        )

        output_folder = os.path.join(
            OUTPUT_DIR,
            category
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        if not os.path.exists(input_folder):

            print(
                f"Folder not found: {input_folder}"
            )

            continue

        video_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        category_count = 0

        print(f"Processing: {category}")

        for video_file in sorted(video_files):

            video_path = os.path.join(
                input_folder,
                video_file
            )

            video_name = os.path.splitext(
                video_file
            )[0]

            cap = cv2.VideoCapture(
                video_path
            )

            if not cap.isOpened():

                print(
                    f"Could not open: {video_file}"
                )

                continue

            frame_number = 0
            video_count = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                if frame_number % FRAME_INTERVAL == 0:

                    filename = (
                        f"{video_name}_"
                        f"frame_{frame_number:06d}.jpg"
                    )

                    output_path = os.path.join(
                        output_folder,
                        filename
                    )

                    cv2.imwrite(
                        output_path,
                        frame
                    )

                    video_count += 1
                    category_count += 1
                    total_frames += 1

                frame_number += 1

            cap.release()

            print(
                f"  {video_file}: "
                f"{video_count} frames"
            )

        print(
            f"Total {category}: "
            f"{category_count} frames\n"
        )

    print("===================================")
    print("FRAME EXTRACTION COMPLETED")
    print("===================================")
    print(
        f"Total frames: {total_frames}"
    )
    print("===================================")


if __name__ == "__main__":
    extract_frames()
    