# Print the supplied video's properties and display its first frame.

from pathlib import Path

import cv2
import numpy as np


video_path = Path(__file__).resolve().parent / "assets" / "small.mp4"

if not video_path.is_file():
    raise FileNotFoundError(f"Video not found: {video_path}")

capture = cv2.VideoCapture(str(video_path))

try:
    if not capture.isOpened():
        raise RuntimeError("Could not open the video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    success, frame = capture.read()

    if not success:
        raise RuntimeError("Could not read the first frame.")

    height, width, channels = frame.shape

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Reported frame count: {frame_count}")
    print(f"Frame shape: {frame.shape}")
    print(f"Pixel data type: {frame.dtype}")

finally:
    capture.release()

print("Click the image window and press any key to close it.")

cv2.imshow("First frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()