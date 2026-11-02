# Read the supplied camera calibration and explain the values needed for PnP.

import argparse
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "assets" / "calibration.yaml"


def load_calibration(path):
    # safe_load reads the YAML settings as ordinary Python data.
    with Path(path).open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError("Calibration must contain a mapping of camera settings.")

    camera_matrix = np.asarray(settings["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(settings["distort_coeffs"], dtype=np.float64).reshape(-1, 1)
    image_size = (int(settings["image_width"]), int(settings["image_height"]))

    if camera_matrix.shape != (3, 3):
        raise ValueError("The camera matrix must have shape (3, 3).")
    if distortion.size not in (4, 5, 8, 12, 14):
        raise ValueError("Unsupported number of OpenCV distortion coefficients.")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration values must be finite.")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("Calibration image dimensions must be positive.")

    return camera_matrix, distortion, image_size


def main():
    parser = argparse.ArgumentParser(description="Inspect the supplied camera calibration.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()

    camera_matrix, distortion, image_size = load_calibration(args.calibration)
    width, height = image_size

    print(f"Calibration file: {args.calibration.name}")
    print(f"Calibrated image resolution: {width} x {height}")
    print("\nCamera matrix K:")
    print(camera_matrix)

    print(f"\nfx: {camera_matrix[0, 0]:.4f} pixels (horizontal focal length)")
    print(f"fy: {camera_matrix[1, 1]:.4f} pixels (vertical focal length)")
    print(f"cx: {camera_matrix[0, 2]:.4f} pixels (principal point x)")
    print(f"cy: {camera_matrix[1, 2]:.4f} pixels (principal point y)")

    print("\nDistortion coefficients:")
    print(distortion.ravel())
    if distortion.size == 5:
        print("Order: k1, k2, p1, p2, k3")

    print("\nUse this calibration with corners restored to the original image resolution.")
    print("Next: combine it with known plate dimensions and the detected corners for PnP.")


if __name__ == "__main__":
    main()
