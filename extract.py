import cv2
import os

# Automatically find the first video file in this folder
exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
videos = [f for f in os.listdir(".") if f.lower().endswith(exts)]
if not videos:
    raise IOError("No video found. Put a video file in this folder.")

video_path = videos[0]
print("Using video:", video_path)

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
step = int(round(fps/2))
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % step == 0:
        sec = count // step
        cv2.imwrite(f"{output_dir}/frame_{sec:04d}.jpg", frame)
        print(f"Saved second {sec}")
    count += 1

cap.release()
print("Done!")