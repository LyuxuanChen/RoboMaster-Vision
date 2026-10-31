# Decode one video frame and use NMS to remove overlapping duplicate predictions.

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from decode_frame import CONFIDENCE_THRESHOLD, decode_predictions
from infer_frame import prepare_frame


NMS_IOU_THRESHOLD = 0.5
ASSETS = Path(__file__).resolve().parent / "assets"


def bounding_box(detection):
    # NMS uses an axis-aligned box enclosing the four original-image corners.
    corners = np.asarray(detection["corners_px"])
    return (
        float(corners[:, 0].min()), float(corners[:, 1].min()),
        float(corners[:, 0].max()), float(corners[:, 1].max()),
    )


def intersection_over_union(box_a, box_b):
    # Boxes contain (left, top, right, bottom), measured in pixels.
    overlap_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    overlap_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = overlap_width * overlap_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD):
    # Sort by confidence, highest first. The input list remains unchanged.
    remaining = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        best_box = bounding_box(best)

        # Suppress duplicates regardless of their predicted colour/pattern.
        # Keep the winning prediction's original corners and label unchanged.
        remaining = [
            candidate for candidate in remaining
            if intersection_over_union(best_box, bounding_box(candidate)) <= iou_threshold
        ]

    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ASSETS / "small.mp4")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.frame_index < 0:
        parser.error("--frame-index must be zero or greater")

    model_path = ASSETS / "0526.onnx"
    for path in (args.video, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        for index in range(args.frame_index + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read frame {index}.")
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

    decoded = decode_predictions(output[0], scale, left, top)
    kept = non_maximum_suppression(decoded)

    print(f"Video: {args.video.name}; frame: {args.frame_index}")
    print(f"Raw candidates: {output.shape[1]}")
    print(f"After confidence filtering ({CONFIDENCE_THRESHOLD}): {len(decoded)}")
    print(f"After NMS (IoU threshold {NMS_IOU_THRESHOLD}): {len(kept)}")

    preview = frame.copy()
    drawing_colors = {
        "blue": (255, 0, 0), "red": (0, 0, 255),
        "neutral": (180, 180, 180), "purple": (255, 0, 255),
    }

    for detection in kept:
        label = (
            f"{detection['color']} {detection['pattern']} "
            f"{detection['confidence']:.4f}"
        )
        print(f"Row {detection['candidate_index']}: {label}")
        points = np.rint(detection["corners_px"]).astype(np.int32)
        color = drawing_colors[detection["color"]]
        cv2.polylines(preview, [points], True, color, 2)
        for point in points:
            cv2.circle(preview, tuple(point), 3, color, -1)
        position = (int(points[:, 0].min()), int(points[:, 1].min()) - 8)
        cv2.putText(
            preview, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )

    if not args.no_display:
        print("Click the image window and press any key to close it.")
        try:
            cv2.namedWindow("Detections after NMS", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detections after NMS", 960, 720)
            cv2.imshow("Detections after NMS", preview)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    return kept


if __name__ == "__main__":
    main()
