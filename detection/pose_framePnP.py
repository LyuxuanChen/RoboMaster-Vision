# Estimate the 3D pose of detected plates and compare projected and observed corners.

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from decode_frame import PATTERNS, decode_predictions
from infer_frame import prepare_frame
from inspect_calibration import DEFAULT_CALIBRATION, load_calibration
from nms_frame import ASSETS, non_maximum_suppression


def physical_corners(pattern):
    # Choose dimensions from the predicted pattern, never from the video name.
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown plate pattern: {pattern}")
    width = 0.230 if pattern in ("1", "base_big") else 0.135
    height = 0.056

    # Plate coordinates in metres: origin at centre, +X right, +Y down.
    # Keep the same corner order as the detector, even when the plate rotates.
    return np.array([
        [-width / 2, -height / 2, 0.0],  # upper-left
        [-width / 2, +height / 2, 0.0],  # lower-left
        [+width / 2, +height / 2, 0.0],  # lower-right
        [+width / 2, -height / 2, 0.0],  # upper-right
    ], dtype=np.float64)


def estimate_pose(corners_px, pattern, camera_matrix, distortion, max_error_px=5.0):
    image_points = np.asarray(corners_px, dtype=np.float64)
    if image_points.shape != (4, 2) or not np.isfinite(image_points).all():
        return None
    if abs(cv2.contourArea(image_points.astype(np.float32))) < 1e-6:
        return None

    object_points = physical_corners(pattern)

    try:
        # IPPE supports a flat rectangle. IPPE_SQUARE would assume a square.
        solutions = cv2.solvePnPGeneric(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None

    if not solutions[0]:
        return None

    best_pose = None
    for rvec, tvec in zip(solutions[1], solutions[2]):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue

        try:
            rotation, _ = cv2.Rodrigues(rvec)

            # Transform EACH physical corner into camera coordinates.
            # tvec alone is the plate centre, not the four corners.
            corners_camera = (rotation @ object_points.T).T + tvec.reshape(1, 3)
            if not np.isfinite(corners_camera).all() or np.any(corners_camera[:, 2] <= 0):
                continue

            # Predict where this 3D rectangle would appear in the original image.
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, distortion
            )
        except cv2.error:
            continue

        projected = projected.reshape(4, 2)
        differences = projected - image_points
        error_px = float(np.sqrt(np.mean(np.sum(differences ** 2, axis=1))))
        if not np.isfinite(error_px):
            continue

        if best_pose is None or error_px < best_pose["reprojection_error_px"]:
            best_pose = {
                "rotation_vector_rad": rvec.reshape(3).tolist(),
                "centre_camera_m": tvec.reshape(3).tolist(),
                "corners_camera_m": corners_camera.tolist(),
                "reprojected_corners_px": projected.tolist(),
                "reprojection_error_px": error_px,
            }

    # An initial sanity bound in original-image pixels, configurable on the CLI.
    # A small reprojection error alone does not guarantee the pose is correct.
    if best_pose is None or best_pose["reprojection_error_px"] > max_error_px:
        return None
    return best_pose


def main():
    parser = argparse.ArgumentParser(description="Estimate plate poses in one video frame.")
    parser.add_argument("--video", type=Path, default=ASSETS / "small.mp4")
    parser.add_argument("--model", type=Path, default=ASSETS / "0526.onnx")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-reprojection-error-px", type=float, default=5.0)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.frame_index < 0:
        parser.error("--frame-index must be zero or greater")
    if not np.isfinite(args.max_reprojection_error_px) or args.max_reprojection_error_px <= 0:
        parser.error("--max-reprojection-error-px must be finite and positive")

    for path in (args.video, args.model, args.calibration):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    camera_matrix, distortion, image_size = load_calibration(args.calibration)
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

    if (frame.shape[1], frame.shape[0]) != image_size:
        raise ValueError("Video resolution does not match the supplied calibration.")

    input_tensor, scale, left, top = prepare_frame(frame)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]
    if output.shape != (1, 25200, 22) or not np.isfinite(output).all():
        raise RuntimeError("Unexpected or non-finite model output.")

    decoded = decode_predictions(output[0], scale, left, top)
    detections = non_maximum_suppression(decoded)
    preview = frame.copy()
    results = []

    print(f"Video: {args.video.name}; frame: {args.frame_index}")
    print(f"Detections after NMS: {len(detections)}")
    print("Camera axes: +X right, +Y down, +Z forward. Distances are in metres.")

    for detection in detections:
        result = dict(detection)
        result["corners_camera_m"] = None
        result["reprojection_error_px"] = None
        pose = estimate_pose(
            detection["corners_px"], detection["pattern"], camera_matrix, distortion,
            args.max_reprojection_error_px,
        )
        name = f"{detection['color']} {detection['pattern']}"
        print(f"\n{name}; confidence={detection['confidence']:.4f}")
        print("Physical corners in the plate's own coordinates (m):")
        print(physical_corners(detection["pattern"]))

        observed = np.rint(detection["corners_px"]).astype(np.int32)
        cv2.polylines(preview, [observed], True, (0, 0, 255), 2)

        if pose is None:
            print("PnP failed validation; keep detection with null pose fields.")
            label = f"{name}: PnP failed"
        else:
            result["corners_camera_m"] = pose["corners_camera_m"]
            result["reprojection_error_px"] = pose["reprojection_error_px"]
            print("Plate centre in camera coordinates (m):", np.round(pose["centre_camera_m"], 4))
            print("Rotation vector (radians, not yaw/pitch/roll):", np.round(pose["rotation_vector_rad"], 4))
            print("Four corners in camera coordinates (m):")
            print(np.round(pose["corners_camera_m"], 4))
            print(f"Reprojection RMS error: {pose['reprojection_error_px']:.4f} pixels")
            label = (
                f"{name}: Z={pose['centre_camera_m'][2]:.2f}m "
                f"RMS={pose['reprojection_error_px']:.2f}px"
            )
            projected = np.rint(pose["reprojected_corners_px"]).astype(np.int32)
            for point in projected:
                cv2.drawMarker(preview, tuple(point), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

        position = (int(observed[:, 0].min()), int(observed[:, 1].min()) - 8)
        cv2.putText(
            preview, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
        results.append(result)

    print("\nRed outlines: detected corners. Green crosses: reprojected 3D corners.")
    if not args.no_display:
        print("Click the image window and press any key to close it.")
        try:
            cv2.namedWindow("PnP pose check", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PnP pose check", 960, 720)
            cv2.imshow("PnP pose check", preview)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    main()
