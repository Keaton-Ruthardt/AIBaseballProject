"""
Create a demo video for class presentation
- Side-by-side: Original video (left) + Annotated with boxes (right)
- Overlay: Prediction probability text
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess

def create_side_by_side_demo(original_video, annotated_video, prediction_prob, output_path):
    """
    Create side-by-side comparison video with prediction overlay

    Args:
        original_video: Path to original video
        annotated_video: Path to annotated video with bounding boxes
        prediction_prob: Probability of SAFE (0-1)
        output_path: Where to save the demo video
    """
    print("Creating demo video for class presentation...")

    # Open both videos
    cap_orig = cv2.VideoCapture(str(original_video))
    cap_annot = cv2.VideoCapture(str(annotated_video))

    # Get video properties
    fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Original video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output video will be double width (side-by-side)
    out_width = width * 2
    out_height = height

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))

    # Calculate prediction text
    pred_text = f"PREDICTION: {'SAFE' if prediction_prob > 0.5 else 'OUT'}"
    prob_text = f"Probability: {prediction_prob:.1%}"

    frame_count = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_annot, frame_annot = cap_annot.read()

        if not ret_orig or not ret_annot:
            break

        # Create side-by-side frame
        combined = np.hstack([frame_orig, frame_annot])

        # Add labels at the top
        cv2.putText(combined, "ORIGINAL", (width//2 - 100, 40),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(combined, "ORIGINAL", (width//2 - 100, 40),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(combined, "PLAYER DETECTION", (width + width//2 - 200, 40),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(combined, "PLAYER DETECTION", (width + width//2 - 200, 40),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Add prediction overlay at bottom center
        # Background rectangle for better visibility
        pred_box_height = 120
        pred_box_y = height - pred_box_height - 20
        cv2.rectangle(combined, (out_width//2 - 300, pred_box_y),
                     (out_width//2 + 300, pred_box_y + pred_box_height),
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(combined, (out_width//2 - 300, pred_box_y),
                     (out_width//2 + 300, pred_box_y + pred_box_height),
                     (255, 255, 255), 3)  # White border

        # Prediction text
        color = (0, 255, 0) if prediction_prob > 0.5 else (0, 0, 255)  # Green for SAFE, Red for OUT
        cv2.putText(combined, pred_text, (out_width//2 - 150, pred_box_y + 50),
                   cv2.FONT_HERSHEY_BOLD, 1.5, color, 3, cv2.LINE_AA)

        # Probability text
        cv2.putText(combined, prob_text, (out_width//2 - 120, pred_box_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Write frame
        out.write(combined)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    # Release everything
    cap_orig.release()
    cap_annot.release()
    out.release()

    print(f"[OK] Demo video created: {output_path}")
    print(f"     Total frames: {frame_count}")
    print(f"     Resolution: {out_width}x{out_height}")


def main():
    print("="*80)
    print("CREATING CLASS DEMO VIDEO - sac_fly_002")
    print("="*80)

    # Define paths
    video_name = "sac_fly_002"
    original_video = Path("DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_002.mp4")
    metadata_file = Path("DataBatch2-20251112T182025Z-1-001/DataBatch2/video_metadata.csv")
    output_folder = Path("demo_output")
    output_folder.mkdir(exist_ok=True)

    print("\n[1/3] Running pipeline on sac_fly_002...")

    # Run the pipeline
    cmd = [
        "venv_deliverable4/Scripts/python",
        "run_complete_pipeline.py",
        "--video", str(original_video),
        "--metadata", str(metadata_file),
        "--output", str(output_folder)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Pipeline failed!")
        print(result.stderr)
        return

    print("   [OK] Pipeline complete")

    # Read prediction probability
    print("\n[2/3] Reading prediction results...")
    pred_file = output_folder / f"{video_name}_prediction.txt"

    prediction_prob = None
    with open(pred_file, 'r') as f:
        for line in f:
            if line.startswith('Probability:'):
                prediction_prob = float(line.split(':')[1].strip())
                break

    if prediction_prob is None:
        print("[ERROR] Could not find prediction probability")
        return

    pred_label = "SAFE" if prediction_prob > 0.5 else "OUT"
    print(f"   Prediction: {pred_label}")
    print(f"   Probability: {prediction_prob:.1%}")

    # Create demo video
    print("\n[3/3] Creating side-by-side demo video...")
    annotated_video = output_folder / f"{video_name}_annotated.mp4"
    demo_video = output_folder / f"{video_name}_CLASS_DEMO.mp4"

    create_side_by_side_demo(
        original_video=original_video,
        annotated_video=annotated_video,
        prediction_prob=prediction_prob,
        output_path=demo_video
    )

    print("\n" + "="*80)
    print("DEMO VIDEO READY!")
    print("="*80)
    print(f"\nFile: {demo_video}")
    print(f"Prediction: {pred_label} ({prediction_prob:.1%})")
    print(f"\nShow this video in class - it has:")
    print(f"  - Original video (left side)")
    print(f"  - Player detection boxes (right side)")
    print(f"  - ML prediction overlay at bottom")


if __name__ == '__main__':
    main()
