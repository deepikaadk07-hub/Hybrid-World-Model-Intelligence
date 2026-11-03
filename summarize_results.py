import pandas as pd

df = pd.read_csv("evaluation_results.csv")

summary = df.groupby("folder_label").agg(
    Videos=("video", "count"),
    Total_Frames=("frames", "sum"),
    Avg_Person_Detection=("person_frame_percentage", "mean"),
    Avg_Inside_Zone=("inside_zone_frame_percentage", "mean"),
    Avg_Inference_Time=("mean_yolo_inference_ms", "mean")
).round(2)

print(summary)

summary.to_csv("category_summary.csv")
print("\nSaved to category_summary.csv")