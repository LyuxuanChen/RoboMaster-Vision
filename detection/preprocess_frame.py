# Prepare the first video frame as a 640 x 640 RGB float16 model input.

from pathlib import Path

import cv2
import numpy as np


def main():
    video_path = Path(__file__).resolve().parent / "assets" / "small.mp4"

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open the video.")

        success, frame = capture.read()

        if not success:
            raise RuntimeError("Could not read the first frame.")
    finally:
        capture.release()

    height, width = frame.shape[:2]

    # Resize without stretching the image.
    scale = min(640 / width, 640 / height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(frame, (new_width, new_height))

    # Add grey padding to centre the resized image in a 640 x 640 square.
    left = (640 - new_width) // 2
    right = 640 - new_width - left
    top = (640 - new_height) // 2
    bottom = 640 - new_height - top

    letterboxed = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # Convert BGR to RGB and scale pixel values from 0-255 to 0-1.
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0

    # Change (height, width, channels) to (channels, height, width).
    channels_first = normalized.transpose(2, 0, 1)

    # Add the batch dimension and match the model's float16 input type.
    input_tensor = np.ascontiguousarray(
        channels_first[np.newaxis, ...], dtype=np.float16
    )

    # Keep the scale and padding for mapping future predictions back.
    print(f"Original resolution: {width} x {height}")
    print(f"Resized resolution: {new_width} x {new_height}")
    print(f"Resize scale: {scale}")
    print(f"Padding: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"Model input shape: {input_tensor.shape}")
    print(f"Model input type: {input_tensor.dtype}")
    print(f"Value range: {input_tensor.min()} to {input_tensor.max()}")
    print("Click the image window and press any key to close it.")

    try:
        cv2.imshow("Prepared image with padding", letterboxed)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
