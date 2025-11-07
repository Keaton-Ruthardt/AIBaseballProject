import cv2

# Step 1: Load the video
video_path = "sac_fly_001.mp4"
cap = cv2.VideoCapture(video_path)

# Step 2: Check if it opened correctly
if not cap.isOpened():
    print("❌ Could not open video file.")
    exit()

# Step 3: Print basic information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✅ OpenCV loaded successfully!")
print(f"Video: {video_path}")
print(f"Resolution: {width}x{height}")
print(f"Frames per second (FPS): {fps:.2f}")
print(f"Total frames: {frame_count}")

# Step 4: Read and save the first frame to confirm success
ret, frame = cap.read()
if ret:
    cv2.imwrite("first_frame.jpg", frame)
    print("🖼️ Saved 'first_frame.jpg' as a sample frame.")
else:
    print("⚠️ Could not read a frame from the video.")

cap.release()
print("✅ Test complete. Video file closed.")
