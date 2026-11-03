from pathlib import Path

root = Path("datasets/restricted_zone_frames")
categories = ["outside", "entering", "staying_inside", "leaving"]

for category in categories:
    folder = root / category
    images = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print(f"{category}: {len(images)} frames")

    if not images:
        print(f"WARNING: No frames found in {category}")

review = root / "reviewLater_frames"

if review.exists():
    for category in ["entering", "leaving"]:
        folder = review / category
        count = sum(
            1 for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"reviewLater_frames/{category}: {count} frames")