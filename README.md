# RoboMaster Vision

A Python computer vision pipeline that detects RoboMaster armour plates in recorded
match footage and estimates their 3D corner positions relative to the camera.
It combines ONNX inference, duplicate suppression and calibrated planar pose
estimation, exporting frame-by-frame results as JSON Lines for further analysis.

**Author:** 陈吕轩 / Lyuxuan Chen

## What the project does

- Detects armour corners, colour and pattern using a pretrained ONNX model.
- Restores detections to the original image resolution after letterbox preprocessing.
- Removes overlapping predictions with non-maximum suppression (NMS).
- Estimates plate pose with OpenCV PnP and exports all four corners in metres.
- Checks positive depth and reprojection error, retaining failed poses as `null`.
- Processes every video frame, including frames without detections, and writes results
  through a temporary file so a failed run preserves any previous completed output.

The implemented work is the **detection and pose-estimation pipeline** in `detection/`.
The repository also retains a separate Rust aiming simulator from the source template;
it is independent of this pipeline and does not contain a completed aiming solution.

## Quick start

Development used **Python 3.11 on macOS**, with CPU inference. The detector weights,
two sample videos and camera calibration are included in `detection/assets/`.
No model training or robot hardware is needed to run the supplied examples.

```bash
git clone https://github.com/LyuxuanChen/RoboMaster-Vision.git
cd RoboMaster-Vision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r detection/requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The main libraries are NumPy, OpenCV, ONNX Runtime and PyYAML; recorded versions
are in [detection/requirements.txt](detection/requirements.txt).

Run these commands **from the repository root** to process the sample clips:

```bash
python detection/run_video_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/results/small.jsonl

python detection/run_video_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/large.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/results/large.jsonl
```

Each command processes one video and creates the output directory if needed.
Existing output is replaced after successful processing; add `--no-overwrite` to
protect it, or choose a different output path. The sample results are also committed
in [detection/results/](detection/results/).

For PyCharm, choose `.venv/bin/python` as the interpreter, set the repository root
as the working directory, select `detection/run_video_detection.py` as the script,
and put the four path arguments above in the Parameters field.

## How it works

```text
Video frame → Letterbox + RGB normalization → ONNX inference
            → Decode labels and corners → Restore pixel coordinates → NMS
            → Calibrated PnP → Pose validation → JSONL record
```

1. **Preprocess.** Preserve aspect ratio while resizing into a 640 × 640 image,
   with padding value 114. For the supplied 1440 × 1080 clips, this gives a
   640 × 480 image with 80-pixel top and bottom padding. Convert BGR to RGB,
   normalize to `[0, 1]` and create a float16 tensor of shape `[1, 3, 640, 640]`.
2. **Infer and decode.** Reuse one CPU ONNX Runtime session. The model returns
   `[1, 25200, 22]` float32 predictions: eight corner coordinates, one objectness
   logit, four colour logits and nine pattern logits. Confidence is
   `sigmoid(objectness)`; colour and pattern use the largest corresponding logits.
3. **Restore and suppress.** Undo padding and scale to recover original-image
   coordinates. Apply greedy, class-agnostic NMS using axis-aligned rectangles
   enclosing each plate. Retained predictions keep their original corners and labels.
4. **Solve pose.** Choose physical geometry from the predicted pattern, then call
   `solvePnPGeneric` with `SOLVEPNP_IPPE` using the original camera calibration.
   Patterns `1` and `base_big` use a 0.230 × 0.056 m plate; other supported patterns
   use 0.135 × 0.056 m. Dimensions describe lightbar endpoints.
5. **Validate and export.** Among finite poses with all corners in front of the
   camera, choose the lowest reprojection RMS error and reject it if it exceeds
   the configured limit. Apply both rotation and translation to each physical corner.
   Failed poses retain the 2D detection with `null` pose fields.

Corner order stays **upper-left, lower-left, lower-right, upper-right**, relative
to the plate's orientation, throughout preprocessing, PnP and export.

### Configuration

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model` | Required | ONNX detector path |
| `--video` | Required | Input video path |
| `--calibration` | Required | Camera calibration YAML |
| `--output` | Required | Destination JSONL path |
| `--confidence-threshold` | `0.5` | Minimum sigmoid(objectness) |
| `--nms-iou-threshold` | `0.5` | Suppress duplicate boxes above this IoU |
| `--max-reprojection-error-px` | `5.0` | Maximum accepted four-corner reprojection RMS |
| `--overwrite` / `--no-overwrite` | Overwrite enabled | Control replacement of existing results |

```bash
python detection/run_video_detection.py --help
```

The current video runner requires **30 FPS** and a frame resolution matching the
calibration. To use other footage, supply calibration for that camera and resolution;
other frame rates require changing the runner's timestamp and FPS handling. Other
detectors require adapting preprocessing, tensor validation and decoding to their
model contract. Plate size is selected from predictions, never from the video filename.

## Output format

Each line is one JSON object with `frame_index`, `timestamp_s` and `detections`.
Frame indices start at zero; timestamps are `frame_index / 30` and restart for each
video. A frame with no detections is still written:

```json
{"frame_index": 0, "timestamp_s": 0.0, "detections": []}
```

| Detection field | Meaning |
| --- | --- |
| `color` | `blue`, `red`, `neutral` or `purple` |
| `pattern` | `sentry`, `1`, `2`, `3`, `4`, `5`, `outpost`, `base_small` or `base_big` |
| `confidence` | Sigmoid(objectness), without multiplying by colour or pattern scores |
| `corners_px` | Four `[x, y]` points in original-image pixels |
| `corners_camera_m` | Four `[X, Y, Z]` points in metres, or `null` if pose validation fails |
| `reprojection_error_px` | Four-corner Euclidean RMS error in pixels, or `null` if pose validation fails |

Camera coordinates use **+X right, +Y down, +Z forward**, with the optical centre as
origin. Both corner arrays use the same order. Exported 3D points are the individual
corners, rather than repeated plate-centre coordinates. Serialization rejects `NaN`
and `Infinity`.

## Repository structure

```text
detection/
  run_video_detection.py      Main video-processing CLI
  infer_frame.py              Letterbox preprocessing and inference demo
  decode_frame.py             Prediction decoding and coordinate restoration
  nms_frame.py                Duplicate suppression and frame demo
  inspect_calibration.py     Camera calibration loader and inspection
  pose_framePnP.py            Physical geometry, PnP and reprojection checks
  validation_debug/           Standalone model, video and preprocessing checks
  assets/                     Pretrained model, sample clips and calibration
  results/                    Exported sample JSONL files
  requirements.txt            Recorded Python dependencies
aiming/                       Independent Rust simulator supplied by the template
vision_development_log.md     Development checkpoints and validation notes
```

Keep the shared Python modules together: the CLI imports them from `detection/`.
Their demonstration entry points run only when executed directly. To inspect a
single frame's pose without opening a window:

```bash
python detection/pose_framePnP.py \
  --video detection/assets/small.mp4 --frame-index 0 --no-display
```

Omit `--no-display` to show detected outlines and reprojected corners. Standalone
inspection tools can also be run from the repository root:

```bash
python detection/validation_debug/inspect_model.py
python detection/validation_debug/inspect_video.py
python detection/validation_debug/preprocess_frame.py
```

The last two commands open preview windows; focus the image and press a key to close.

## Sample results and limits

The committed results at the default thresholds contain:

| Video | Frame records | Per-frame detections | Empty frames | Failed poses retained |
| --- | ---: | ---: | ---: | ---: |
| `small.mp4` | 360 | 395 | 162 | 0 |
| `large.mp4` | 360 | 150 | 260 | 3 |

These totals count predictions across frames, not unique tracked plates. The
[development log](vision_development_log.md) records checks of ordering, timestamps,
output fields, finite coordinates, physical dimensions, reprojection consistency,
NMS overlap and synthetic poses.

This is an offline detection pipeline. Cross-frame tracking, motion prediction,
ballistic aiming and hardware control are outside the implemented scope. Thresholds
have not been optimized against ground-truth annotations, and low reprojection
error alone does not establish real-world distance accuracy. The sample counts are
output summaries, not precision, recall or real-time performance benchmarks.

## Credits and provenance

Developed from the [HKU Astar vision template](https://github.com/HKUAstar/vision-training-assignment),
which supplied the recordings, calibration, model and independent aiming simulator.
The pretrained detector originates from
[RobotPilots / RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel).
My contribution is the Python preprocessing, inference integration, decoding, NMS,
PnP validation, JSONL export and inspection workflow described here. Implementation
and explanations were developed with AI assistance, as recorded in the development log.
