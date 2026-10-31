# RoboMaster Vision

Python implementation of the **armour detection and pose estimation** route of the
[HerKules vision assignment](https://training.herkules.dev/learn/vision-assignment).
Development uses PyCharm, a project virtual environment and CPU inference.

The current implementation reads a selected frame, prepares it for the supplied ONNX
model, decodes predictions, restores corner coordinates and removes duplicates with
non-maximum suppression (NMS). **PnP, 3D corner estimation and full-video JSONL export
are still to be implemented.**

## Setup

Developed with Python 3.11 on macOS. Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r detection/requirements.txt
```

If the environment already exists, reuse it rather than recreating it. In PyCharm,
select `.venv/bin/python` as the project interpreter.
The main packages are NumPy, OpenCV, ONNX Runtime and PyYAML; exact installed versions
are in `detection/requirements.txt`.

The supplied assets should be present in `detection/assets/`:

```text
0526.onnx
small.mp4
large.mp4
calibration.yaml
```

## Run the current detector

These commands assume the project environment is active and the terminal is at the
repository root.

Display the first frame of the small video with detections after NMS:

```bash
python detection/nms_frame.py
```

Inspect the other video or a different frame:

```bash
python detection/nms_frame.py --video detection/assets/large.mp4
python detection/nms_frame.py --frame-index 180
```

Print results without opening a window:

```bash
python detection/nms_frame.py --no-display
```

Frame indices start at zero. Click the image window and press any key to close it.
Each run processes only the selected frame and does not yet write a JSONL file.

The checked result for the first frame of `small.mp4` is:

```text
Raw candidates: 25200
After confidence filtering (0.5): 19
After NMS (IoU threshold 0.5): 2
Row 16601: red 3 0.9539
Row 3414: red 4 0.8658
```

These are model predictions on a sample frame, not ground-truth accuracy measurements.

## Learning checkpoints

Earlier scripts are retained to show how the implementation developed:

| Script in `detection/` | Purpose |
| --- | --- |
| `inspect_model.py` | Load the model and inspect tensor metadata |
| `inspect_video.py` | Read video properties and display the first frame |
| `preprocess_frame.py` | Resize, pad, convert colours and prepare model input |
| `infer_frame.py` | Run inference and inspect the highest-objectness candidate |
| `decode_frame.py` | Decode confident candidates and draw them before NMS |
| `nms_frame.py` | Suppress duplicates and inspect a selected frame |

Run checkpoints with `python detection/<script-name>.py`.
The decoding and NMS scripts reuse helpers from earlier scripts, so keep these files
in the `detection/` folder.

See [vision_development_log.md](vision_development_log.md) for milestones, checks and
remaining work. Day numbers refer to planned learning milestones; several were worked
on in the same calendar day.

## Current processing choices

- Preserve aspect ratio when resizing to fit 640 x 640 pixels; centre with padding value 114.
- Convert BGR to RGB, normalize to `[0, 1]`, and provide float16 input of shape `[1, 3, 640, 640]`.
- Decode float32 output of shape `[1, 25200, 22]`.
- Use sigmoid(objectness) as confidence and retain values at least `0.5`.
- Select colour and pattern by the largest logits, using the assignment's category tables.
- Undo padding and resizing to restore original-image pixel coordinates.
- Preserve the lightbar endpoint order: upper-left, lower-left, lower-right, upper-right,
  relative to the plate's orientation.
- Use class-agnostic NMS with IoU threshold `0.5` on enclosing axis-aligned boxes.
  The retained prediction keeps its quadrilateral and label.

Checks cover frames 0 and 180 of both clips, plus basic IoU and NMS cases.
Full-video validation remains pending.

## Remaining assignment work

- Load calibration and select plate dimensions from the predicted pattern.
- Estimate pose, recover 3D corners in the camera frame and calculate reprojection error.
- Process every frame and export the required JSONL, including empty frames and failed poses.
- Add model, calibration and output path arguments to the final program.
- Check reproducibility from a fresh environment and complete submission validation.

## References and assistance

- Assignment requirements: [English](detection/README.en.md) / [Chinese](detection/README.md).
- Supplied pretrained model: [RobotPilots / RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel).
- Original template overview: [README.en.md](README.en.md).

AI assistance was used to develop and explain the learning scripts and documentation.
The model is pretrained; these scripts perform inference rather than training.
