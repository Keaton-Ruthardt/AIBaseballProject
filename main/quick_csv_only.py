"""
Quick CSV generation - no video output
"""

from simple_tracker import SimpleOutfielderTracker
import logging

logging.basicConfig(level=logging.INFO)

tracker = SimpleOutfielderTracker(model_size='n', confidence_threshold=0.25, device='cpu')

# Process only frames 300-420
results = tracker.process_video(
    video_path="week1_deliverables/baseballVideos/sac_fly_002.mp4",
    output_path=None,  # No video output
    start_frame=300,
    end_frame=420
)

# Export CSV
df = tracker.export_simple_csv(results, "simple_output.csv")

print("\n=== SAMPLE OUTPUT ===")
print(df.head(10).to_string())

print("\n=== CATCH EVENTS ===")
catches = df[df['catcher_player_id'] != -1]
print(catches[['frame_number', 'catcher_player_id', 'catch_distance_px']].to_string(index=False))
