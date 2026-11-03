
import pandas as pd
from pathlib import Path

file_path = Path("baseline_results.csv")

if not file_path.exists():
    raise FileNotFoundError("Run compare_baseline.py first.")

df = pd.read_csv(file_path)

summary = df.groupby("category").agg(
    Videos=("video", "count"),
    Total_Frames=("frames", "sum"),
    Avg_Detection_Percent=("detection_frame_percent", "mean"),
    Avg_Inside_Zone_Percent=("inside_zone_frame_percent", "mean"),
    Avg_Processing_Time_ms=("avg_processing_ms", "mean"),
    Avg_FPS=("fps", "mean"),
    Avg_Unique_Track_IDs=("unique_track_ids", "mean")
).round(2)

print("\nBASELINE PERFORMANCE SUMMARY")
print(summary.to_string())

summary.to_csv("baseline_category_summary.csv")
print("\nSaved: baseline_category_summary.csv")
