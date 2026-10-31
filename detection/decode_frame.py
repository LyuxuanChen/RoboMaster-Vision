# Decode confident armour candidates and draw their corners on the original frame.
# This learning checkpoint does not remove duplicate predictions yet.

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from infer_frame import prepare_frame


# Use the assignment's category order, which differs from the upstream README.
COLORS = ["blue", "red", "neutral", "purple"]
PATTERNS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"]
CONFIDENCE_THRESHOLD = 0.5


def sigmoid(logit):
    # Both branches calculate sigmoid while avoiding exponential overflow.
    if logit >= 0:
        return 1.0 / (1.0 + np.exp(-logit))
    exponential = np.exp(logit)
    return exponential / (1.0 + exponential)


def decode_predictions(predictions, scale, left, top, threshold=CONFIDENCE_THRESHOLD):
    detections = []

    for index, candidate in enumerate(predictions):
        confidence = float(sigmoid(float(candidate[8])))
        if confidence < threshold:
            continue

        color_index = int(np.argmax(candidate[9:13]))
        pattern_index = int(np.argmax(candidate[13:22]))

        # Copy the four points, preserving upper-left, lower-left,
        # lower-right, upper-right order.
        corners = candidate[:8].reshape(4, 2).copy()

        # Undo letterbox padding and resizing to get original-image pixels.
        corners[:, 0] = (corners[:, 0] - left) / scale
        corners[:, 1] = (corners[:, 1] - top) / scale

        detections.append({
            "candidate_index": index,
            "color": COLORS[color_index],
            "pattern": PATTERNS[pattern_index],
            "confidence": confidence,
            "corners_px": corners.tolist(),
        })

    return detections


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
    output = session.run([output_name], {input_name: input_tensor})[0]

    if output.shape != (1, 25200, 22) or not np.isfinite(output).all():
        raise RuntimeError("Unexpected or non-finite model output.")

    detections = decode_predictions(output[0], scale, left, top)

    print(f"Raw candidates: {output.shape[1]}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Candidates kept before NMS: {len(detections)}")

    preview = frame.copy()
    drawing_colors = {
        "blue": (255, 0, 0),
        "red": (0, 0, 255),
        "neutral": (180, 180, 180),
        "purple": (255, 0, 255),
    }

    for detection in detections:
        print(
            f"Row {detection['candidate_index']}: "
            f"{detection['color']} {detection['pattern']}, "
            f"confidence={detection['confidence']:.4f}"
        )
        print("Original-image corners:", np.round(detection["corners_px"], 2))

        points = np.rint(detection["corners_px"]).astype(np.int32)
        color = drawing_colors[detection["color"]]
        cv2.polylines(preview, [points], True, color, 2)
        for point in points:
            cv2.circle(preview, tuple(point), 3, color, -1)

        label = (
            f"{detection['color']} {detection['pattern']} "
            f"{detection['confidence']:.2f}"
        )
        label_position = (int(points[:, 0].min()), int(points[:, 1].min()) - 8)
        cv2.putText(
            preview, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    print("Duplicates may overlap: NMS will be added in the next checkpoint.")
    print("Click the image window and press any key to close it.")
    try:
        cv2.namedWindow("Decoded candidates before NMS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Decoded candidates before NMS", 960, 720)
        cv2.imshow("Decoded candidates before NMS", preview)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    main()
