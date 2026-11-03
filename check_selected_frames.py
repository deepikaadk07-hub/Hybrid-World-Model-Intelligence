from pathlib import Path

root = Path("datasets/restricted_zone_frames")

selected = {
    "entering": [
        "entering_01", "entering_02",
        "entering_04", "entering_09"
    ],
    "leaving": [
        "leaving_04", "leaving_07", "leaving_09"
    ]
}

for category, videos in selected.items():
    print(f"\n{category.upper()}")

    for video in videos:
        matches = [
            f for f in (root / category).rglob("*")
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and video in f.name
        ]

        print(f"{video}: {len(matches)} frames")