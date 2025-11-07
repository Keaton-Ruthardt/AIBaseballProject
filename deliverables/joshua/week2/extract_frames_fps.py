
import cv2, csv, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='path to input video (mp4/avi)')
parser.add_argument('--fps', type=float, default=30.0, help='target FPS to extract')
parser.add_argument('--out', default='out', help='output folder')
parser.add_argument('--jpeg-quality', type=int, default=90, help='JPEG quality 0-100')
args = parser.parse_args()

# Setup
outdir = Path(args.out)
frames_dir = outdir / 'frames'
frames_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise RuntimeError(f'Cannot open video: {args.input}')

native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {args.input}\nResolution: {width}x{height}\nNative FPS: {native_fps:.2f}\nFrames: {frame_count}")

# Time-based sampling using frame index → timestamp
# (For VFR videos, OpenCV timestamps can be unreliable; this is a simple, robust baseline.)
step = 1 if native_fps == 0 else max(1, round(native_fps / args.fps))
print(f"Target FPS: {args.fps:.2f}  |  Approx. step: every {step} frame(s)")

csv_path = outdir / 'frames.csv'

start_time = time.time()
saved = 0
idx = 0

# JPEG params
params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_idx', 'time_seconds', 'path'])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            # Compute timestamp from frame index
            t = idx / (native_fps if native_fps > 0 else args.fps)
            out_path = frames_dir / f'frame_{idx:06d}.jpg'
            cv2.imwrite(str(out_path), frame, params)
            writer.writerow([idx, f"{t:.3f}", str(out_path)])
            saved += 1
        idx += 1

cap.release()
elapsed = time.time() - start_time
print(f"Saved {saved} frames → {frames_dir}")
print(f"Elapsed: {elapsed:.2f}s  |  Proc rate: {saved/max(elapsed,1e-6):.1f} frames/sec")
