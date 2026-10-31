# Prepare one video frame, run the ONNX model, and inspect its raw predictions.

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def prepare_frame(frame):
    # Repeat the preprocessing from preprocess_frame.py so that earlier
    # learning checkpoints can stay unchanged.
    height, width = frame.shape[:2]
    scale = min(640 / width, 640 / height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(frame, (new_width, new_height))

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

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    channels_first = normalized.transpose(2, 0, 1)
    input_tensor = np.ascontiguousarray(
        channels_first[np.newaxis, ...], dtype=np.float16
    )

    return input_tensor, scale, left, top


def main():
    assets = Path(__file__).resolve().parent / "assets"
    video_path = assets / "small.mp4"
    model_path = assets / "0526.onnx"

    for path in (video_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open the video.")
        success, frame = capture.read()
        if not success:
            raise RuntimeError("Could not read the first frame.")
    finally:
        capture.release()

    input_tensor, scale, left, top = prepare_frame(frame)

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Unlike get_inputs()/get_outputs(), run() actually processes the image.
    output = session.run([output_name], {input_name: input_tensor})[0]

    if output.shape != (1, 25200, 22):
        raise RuntimeError(f"Unexpected output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("The model returned NaN or infinite values.")

    # Remove the batch dimension to get one row per candidate.
    predictions = output[0]

    print("Inference completed on the first frame of small.mp4!")
    print(f"Input shape: {input_tensor.shape}; type: {input_tensor.dtype}")
    print(f"Output shape: {output.shape}; type: {output.dtype}")
    print(f"Candidate predictions: {len(predictions)}")
    print(f"Resize scale: {scale}; padding: left={left}, top={top}")

    # This is a candidate for inspection, not a final filtered detection.
    best_index = int(np.argmax(predictions[:, 8]))
    candidate = predictions[best_index]
    print(f"\nCandidate with the highest objectness logit: row {best_index}")
    print("Four corners in the 640 x 640 image:")
    print(candidate[:8].reshape(4, 2))
    print(f"Objectness logit: {candidate[8]}")
    print("Colour logits:", candidate[9:13])
    print("Pattern logits:", candidate[13:22])


if __name__ == "__main__":
    main()
