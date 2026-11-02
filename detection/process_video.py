# Process both supplied videos by default and export a separate JSONL for each.

import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import onnxruntime as ort

from decode_frame import decode_predictions
from infer_frame import prepare_frame
from inspect_calibration import load_calibration
from nms_frame import non_maximum_suppression
from pose_framePnP import estimate_pose


# The supplied recordings use a fixed 30 FPS timeline.
FPS = 30.0

# Default locations are relative to this script, so PyCharm's working directory
# does not matter when running without parameters.
DETECTION_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = DETECTION_DIR / "assets" / "0526.onnx"
DEFAULT_VIDEOS = (
    DETECTION_DIR / "assets" / "small.mp4",
    DETECTION_DIR / "assets" / "large.mp4",
)
DEFAULT_CALIBRATION = DETECTION_DIR / "assets" / "calibration.yaml"
DEFAULT_OUTPUT_DIR = DETECTION_DIR / "results"


def detect_frame(frame, session, camera_matrix, distortion, image_size,
                 confidence_threshold=0.5, nms_iou_threshold=0.5, max_error_px=5.0):
    if (frame.shape[1], frame.shape[0]) != image_size:
        raise ValueError("Video resolution does not match the supplied calibration.")

    input_tensor, scale, left, top = prepare_frame(frame)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]
    if output.shape != (1, 25200, 22) or not np.isfinite(output).all():
        raise RuntimeError("Unexpected or non-finite model output.")

    decoded = decode_predictions(output[0], scale, left, top, confidence_threshold)
    kept = non_maximum_suppression(decoded, nms_iou_threshold)
    detections = []

    for candidate in kept:
        pose = estimate_pose(
            candidate["corners_px"], candidate["pattern"],
            camera_matrix, distortion, max_error_px,
        )

        # Write exactly the required detection fields. Keep failed poses as null.
        detections.append({
            "color": candidate["color"],
            "pattern": candidate["pattern"],
            "confidence": candidate["confidence"],
            "corners_px": candidate["corners_px"],
            "corners_camera_m": None if pose is None else pose["corners_camera_m"],
            "reprojection_error_px": None if pose is None else pose["reprojection_error_px"],
        })

    return detections


def process_one_video(video_path, output_path, session, camera_matrix, distortion,
                      image_size, confidence_threshold=0.5, nms_iou_threshold=0.5,
                      max_error_px=5.0, overwrite=True):
    print(f"\nProcessing {video_path.name}...", flush=True)
    capture = cv2.VideoCapture(str(video_path))
    temporary_path = None

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(reported_fps) or not np.isclose(reported_fps, FPS, atol=0.01, rtol=0):
            raise ValueError(f"Expected a supplied 30 FPS clip; video reports {reported_fps} FPS.")
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        detection_count = 0
        empty_frames = 0
        failed_poses = 0

        # Write to a temporary file first so a failed run does not publish partial results.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)

            while True:
                success, frame = capture.read()
                if not success:
                    break

                detections = detect_frame(
                    frame, session, camera_matrix, distortion, image_size,
                    confidence_threshold, nms_iou_threshold, max_error_px,
                )
                record = {
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / FPS,
                    "detections": detections,
                }

                # One complete JSON object per line; Python None becomes JSON null.
                # allow_nan=False rejects non-standard NaN and Infinity values.
                output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

                frame_index += 1
                detection_count += len(detections)
                empty_frames += int(not detections)
                failed_poses += sum(item["corners_camera_m"] is None for item in detections)
                if frame_index % 60 == 0:
                    print(f"Processed {frame_index} frames...", flush=True)

        if frame_index == 0:
            raise RuntimeError("The video did not contain any readable frames.")
        if reported_count > 0 and frame_index != reported_count:
            raise RuntimeError(
                f"Decoded {frame_index} frames, but video metadata reports {reported_count}."
            )
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        temporary_path.replace(output_path)
        temporary_path = None

        print(f"Saved {frame_index} JSONL records to {output_path}")
        print(f"Detections across all frames: {detection_count}")
        print(f"Frames with no detections: {empty_frames}")
        print(f"Detections with failed poses (retained with null fields): {failed_poses}")
    finally:
        capture.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export both supplied videos, or one video selected with --video."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, help="Select one video instead of the default pair.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, help="Output file for a single --video run.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-reprojection-error-px", type=float, default=5.0)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=True,
        help="Replace generated output on reruns (default); use --no-overwrite to keep existing output.",
    )
    args = parser.parse_args()

    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between 0 and 1")
    if not 0 <= args.nms_iou_threshold <= 1:
        parser.error("--nms-iou-threshold must be between 0 and 1")
    if not np.isfinite(args.max_reprojection_error_px) or args.max_reprojection_error_px <= 0:
        parser.error("--max-reprojection-error-px must be finite and positive")
    if args.output is not None and args.video is None:
        parser.error("--output requires --video; omit both to export the default pair")

    # With no video argument, prepare one job for each supplied recording.
    videos = DEFAULT_VIDEOS if args.video is None else (args.video,)
    jobs = [
        (video, args.output if args.output is not None
         else DEFAULT_OUTPUT_DIR / f"{video.stem}.jsonl")
        for video in videos
    ]

    input_paths = (args.model, args.calibration, *videos)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    for _, output_path in jobs:
        if any(output_path.resolve() == path.resolve() for path in input_paths):
            parser.error("Output paths must differ from the input paths")
        if output_path.exists() and not args.overwrite:
            parser.error(f"Output already exists: {output_path}; rerun with --overwrite to replace it")

    camera_matrix, distortion, image_size = load_calibration(args.calibration)
    # Share the loaded model across both videos. Frame indices and timestamps
    # restart at zero inside process_one_video for each recording.
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    for video_path, output_path in jobs:
        process_one_video(
            video_path, output_path, session, camera_matrix, distortion, image_size,
            args.confidence_threshold, args.nms_iou_threshold,
            args.max_reprojection_error_px, args.overwrite,
        )


if __name__ == "__main__":
    main()
