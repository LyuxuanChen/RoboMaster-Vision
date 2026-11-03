# Detection implementation and run guide

This document explains the Python implementation in this repository. The original
assignment READMEs are preserved. The selected assignment route is **armour detection
and pose estimation**; the `aiming/` directory is not used by this implementation.

The main executable is **`detection/run_video_detection.py`**. It processes one video
per invocation and writes one JSON Lines record for every frame. Run it twice to
produce separate results for the two supplied videos.

## 1. Repository structure

```text
RoboMaster-Vision/
├── README.md                         # Original assignment introduction
├── README.en.md                      # Original English introduction
├── IMPLEMENTATION.md                 # This implementation guide
├── vision_development_log.md         # Development milestones and notes
├── aiming/                           # Separate assignment route; not used here
└── detection/
    ├── run_video_detection.py         # Main command-line executable
    ├── infer_frame.py                 # Shared preprocessing + inference demo
    ├── decode_frame.py                # Shared decoding + pre-NMS demo
    ├── nms_frame.py                   # Shared NMS + sample-frame demo
    ├── inspect_calibration.py         # Shared calibration loader + inspection demo
    ├── pose_framePnP.py               # Shared PnP solver + sample-frame demo
    ├── requirements.txt              # Python dependency versions
    ├── README.md                     # Original detection requirements
    ├── README.en.md                  # Original English detection requirements
    ├── assets/
    │   ├── 0526.onnx                  # Supplied pretrained detector
    │   ├── small.mp4                  # Supplied video, 360 frames at 30 FPS
    │   ├── large.mp4                  # Supplied video, 360 frames at 30 FPS
    │   └── calibration.yaml           # Camera calibration for the original images
    ├── results/
    │   ├── small.jsonl                # Generated small-video results
    │   └── large.jsonl                # Generated large-video results
    └── validation_debug/
        ├── inspect_model.py          # Standalone model-metadata check
        ├── inspect_video.py          # Standalone video-reading check
        └── preprocess_frame.py       # Standalone preprocessing preview
```

The `.venv/` folder contains the local Python environment and is excluded from Git.
Output directories are created automatically when the program needs them.

## 2. Which script does what?

### Main executable and shared modules

| Script | Functions used by the pipeline | Behaviour when run directly |
| --- | --- | --- |
| `run_video_detection.py` | `main()`, `process_one_video()`, `detect_frame()` | Processes all frames of the selected video and exports JSONL |
| `infer_frame.py` | `prepare_frame()` | Runs inference on the first small-video frame and prints the highest-objectness candidate |
| `decode_frame.py` | `decode_predictions()`, which uses `sigmoid()` | Prints and draws all candidates above the confidence threshold before NMS |
| `nms_frame.py` | `non_maximum_suppression()`, which uses `bounding_box()` and `intersection_over_union()` | Displays retained detections on a selected frame |
| `inspect_calibration.py` | `load_calibration()` | Prints camera resolution, matrix and distortion coefficients |
| `pose_framePnP.py` | `estimate_pose()`, which uses `physical_corners()` | Prints pose and 3D corners, and draws a reprojection preview |

These shared modules are required by the executable; do not copy only the main
file when reproducing the program. Keep them together in `detection/`.

The shared modules place their demonstration entry points behind:

```python
if __name__ == "__main__":
    main()
```

Running a module directly calls its demo. Importing a function from it makes that
function available without automatically running the demo or opening its window.

### Standalone validation/debug scripts

The files in `validation_debug/` are manual inspection tools, retained as learning
checkpoints. The complete pipeline does not import them.

| Script | Purpose | Run from repository root |
| --- | --- | --- |
| `inspect_model.py` | Load the model and print input/output metadata | `python detection/validation_debug/inspect_model.py` |
| `inspect_video.py` | Print video properties and display the first frame | `python detection/validation_debug/inspect_video.py` |
| `preprocess_frame.py` | Display letterboxing and print the prepared tensor shape/type | `python detection/validation_debug/preprocess_frame.py` |

These scripts locate their assets relative to their own file locations. Close
preview windows by clicking the image and pressing a key.

## 3. Pipeline

### What gets called during a full run?

```text
run_video_detection.py: main()
  |
  +-- load_calibration()                    [inspect_calibration.py]
  +-- create an ONNX Runtime session once
  +-- process_one_video()
        |
        +-- read the next frame
        +-- detect_frame()
        |     +-- prepare_frame()           [infer_frame.py]
        |     +-- session.run()             [ONNX Runtime]
        |     +-- decode_predictions()      [decode_frame.py]
        |     +-- non_maximum_suppression()  [nms_frame.py]
        |     +-- for each retained plate:
        |           +-- estimate_pose()      [pose_framePnP.py]
        |                 +-- physical_corners()
        |                 +-- solvePnPGeneric(IPPE)
        |                 +-- transform corners and check reprojection
        +-- write one JSON object and a newline
        +-- repeat until the end of the video
```

Some shared modules also import helpers from one another. For example,
`decode_frame.py` imports `prepare_frame()` from `infer_frame.py`.

### Processing choices

1. **Read frames in order.** The supplied clips are 1440 x 1080 at 30 FPS. The model
   session is reused across frames. Frame size is checked against the calibration.
2. **Prepare model input.** Resize proportionally to 640 x 480 and add 80 pixels
   of padding above and below, using value 114. This preserves the full scene and
   its aspect ratio. Convert BGR to RGB, normalize to `[0, 1]`, move channels first
   and add the batch dimension. Input shape is `[1, 3, 640, 640]`, type `float16`.
3. **Run inference.** Output shape is `[1, 25200, 22]`, type `float32`. Rows are
   candidate predictions, not counts of physical plates.
4. **Decode and filter.** Apply sigmoid to objectness and retain scores at least
   `0.5`. Select the largest colour and pattern logits using the assignment's
   category order. Do not multiply those logits into the confidence score.
5. **Restore image coordinates.** Subtract letterbox padding and divide by the
   resize scale. Keep corner order: upper-left, lower-left, lower-right, upper-right,
   relative to the plate's orientation. The points represent lightbar endpoints.
6. **Suppress duplicates.** Apply greedy, class-agnostic NMS using enclosing
   axis-aligned rectangles and IoU threshold `0.5`. The winner keeps its original
   corners and label; conflicting overlapping labels compete for that same location.
7. **Estimate pose.** Patterns `1` and `base_big` use a 0.230 x 0.056 m rectangle;
   the other defined patterns use 0.135 x 0.056 m. Use original-image corners,
   calibration and `SOLVEPNP_IPPE`. Check candidate poses for finite values and
   positive corner depth, then choose the one with the smallest reprojection RMS.
8. **Validate and export.** Reject poses above the initial 5-pixel RMS limit.
   Transform every physical corner using rotation and translation. Keep detections
   with failed poses, setting both pose fields to `null`.

The initial thresholds are configurable and have not been optimized against
ground-truth annotations. Small reprojection error measures agreement with detected
corners; it does not independently establish real-world distance accuracy.

## 4. Full run instructions

### Environment setup

Development used Python 3.11 on macOS with CPU inference. From the repository root,
create an environment if one does not already exist:

```bash
python3.11 -m venv .venv
```

Activate it and install the recorded dependencies:

```bash
source .venv/bin/activate
python -m pip install -r detection/requirements.txt
```

Reuse an existing environment rather than recreating it on every run. In PyCharm,
select the project's `.venv/bin/python` as the interpreter.

The main dependencies are NumPy, OpenCV, ONNX Runtime and PyYAML. The model is
pretrained; this repository runs inference and does not train new model weights.


### From the repository root

The commands in this section assume the terminal is inside `RoboMaster-Vision/`.
Use `pwd` to check. If the terminal is already inside `detection/`, run `cd ..` once
before copying these commands. Quote full directory paths if they contain spaces.

Process the small video:

```bash
python detection/run_video_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/small.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/results/small.jsonl
```

Process the large video:

```bash
python detection/run_video_detection.py \
  --model detection/assets/0526.onnx \
  --video detection/assets/large.mp4 \
  --calibration detection/assets/calibration.yaml \
  --output detection/results/large.jsonl
```

Each command runs the same pipeline with different file paths. The program does
not choose plate dimensions from the video filename.

### From inside the detection folder

If the terminal is inside `RoboMaster-Vision/detection/`, omit the extra `detection/`
prefix from every path:

```bash
python run_video_detection.py \
  --model assets/0526.onnx \
  --video assets/small.mp4 \
  --calibration assets/calibration.yaml \
  --output results/small.jsonl
```

For the large video, change `small.mp4` and `small.jsonl` to `large.mp4` and
`large.jsonl`. Relative paths are interpreted from the terminal's working directory.

### From PyCharm

Open **Run > Edit Configurations**, select the script configuration, and set:

- **Script:** `detection/run_video_detection.py`, selected using the file chooser.
- **Interpreter:** the project virtual environment.
- **Working directory:** the repository root, `RoboMaster-Vision/`.
- **Parameters:** the following arguments, entered on one line:

```text
--model detection/assets/0526.onnx --video detection/assets/small.mp4 --calibration detection/assets/calibration.yaml --output detection/results/small.jsonl
```

Do not include `python` or the script filename in the Parameters field. All four
path parameters are required; running the executable without them prints a usage
message and exits with code 2.

### Command-line arguments

| Argument | Required/default | Purpose |
| --- | --- | --- |
| `--model` | Required | ONNX model path |
| `--video` | Required | Video to process |
| `--calibration` | Required | Camera calibration YAML path |
| `--output` | Required | Destination JSONL file |
| `--confidence-threshold` | `0.5` | Minimum sigmoid(objectness) score to retain |
| `--nms-iou-threshold` | `0.5` | Suppress overlap above this IoU |
| `--max-reprojection-error-px` | `5.0` | Maximum accepted PnP RMS error in original-image pixels |
| `--overwrite` | Enabled by default | Replace an existing generated result after successful processing |
| `--no-overwrite` | Optional | Refuse to replace an existing result |

Show the argument help with:

```bash
python detection/run_video_detection.py --help
```

A temporary file is written during processing. It replaces the destination only
when the video completes successfully, preserving an earlier completed result if
the new run fails. Each run writes fresh records rather than appending them.

## 5. Output format

JSONL means one complete JSON object on each line, with no surrounding array.
Each supplied video produces **360 lines**, indexed 0 through 359. Timestamps are
`frame_index / 30`, restarting at zero for each video.

Each frame has `frame_index`, `timestamp_s` and `detections`. An empty-detection
record has this structure:

```json
{"frame_index": 0, "timestamp_s": 0.0, "detections": []}
```

Each object inside `detections` contains:

| Field | Meaning |
| --- | --- |
| `color` | `blue`, `red`, `neutral` or `purple` |
| `pattern` | A pattern label; numeric labels are strings such as `"3"` |
| `confidence` | Sigmoid(objectness), between 0 and 1 |
| `corners_px` | Four `[x, y]` points in original-image pixels |
| `corners_camera_m` | Four `[X, Y, Z]` points in metres, or `null` when PnP fails |
| `reprojection_error_px` | Four-corner Euclidean RMS error in pixels, or `null` when PnP fails |

Camera coordinates use the camera's optical centre as origin: **+X right, +Y down,
+Z forward**. Both corner arrays preserve the same point order. The exported 3D
values are corner locations, not four copies of the plate centre.

The program rejects non-finite model output and serializes with `allow_nan=False`,
so it does not intentionally write `NaN` or `Infinity` into JSON.

## 6. Checks completed and remaining review

Full-video checks with the default thresholds produced:

| Video | Frame records | Detections across all frames | Empty frames | Failed poses retained |
| --- | ---: | ---: | ---: | ---: |
| `small.mp4` | 360 | 395 | 162 | 0 |
| `large.mp4` | 360 | 150 | 260 | 3 |

Detection totals count per-frame predictions, not unique tracked plates. No
cross-frame tracking is implemented or required.

Checks covered record order, timestamps, required fields, finite coordinates,
physical plate dimensions and consistency between the exported 3D corners and
reprojection error. Additional checks covered empty detections, retained failed
poses, NMS overlap cases and synthetic poses with known geometry.

These checks validate output structure and geometric consistency. Broader visual
review and a clean-environment reproduction check remain part of final submission
preparation. The complete assignment requirements remain in
[detection/README.en.md](detection/README.en.md) and
[detection/README.md](detection/README.md).

See [vision_development_log.md](vision_development_log.md) for development history.
The scripts were developed incrementally with AI assistance and kept as learning
checkpoints alongside the final executable.
