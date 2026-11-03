# Vision development log

Selected route: armour detection and pose estimation, implemented in Python.

Updated: 2026-11-03. Day numbers refer to the seven planned learning milestones,
not separate calendar dates. These entries record the state at each checkpoint.
AI assistance was used for implementation and explanations; verification is recorded below.

## Day 1 — Environment and model inspection

Completed:
- Created a project virtual environment.
- Installed NumPy, OpenCV, ONNX Runtime and PyYAML.
- Saved dependency versions in requirements.txt.
- Created inspect_model.py and successfully loaded 0526.onnx.

Verified:
- Input: [1, 3, 640, 640], float16.
- Output: [1, 25200, 22], float32.

Understanding:
- The script reads model metadata.
- No video frame has been passed through the model yet.

Next:
- Read a video frame and prepare it for inference.

## Day 2 — Video reading, preprocessing and inference

Completed:

- Created `detection/inspect_video.py` to read video properties and display the first frame.
- Created `detection/preprocess_frame.py` to prepare a frame for the model.
- Created `detection/infer_frame.py` to run inference and inspect a candidate.
- Kept earlier scripts as separate learning checkpoints.

Verified:

- `small.mp4` reports 1440 x 1080 pixels, 30 FPS and 360 frames.
- The first frame has shape `(1080, 1440, 3)` and type `uint8`.
- Resizing produces 640 x 480 pixels, with 80 pixels of grey padding above and below.
- BGR is converted to RGB, pixel values are normalized to `[0, 1]`, channels are moved
  first, and a batch dimension is added.
- Input shape is `(1, 3, 640, 640)`, type `float16`.
- `session.run()` returns finite predictions with shape `(1, 25200, 22)`, type `float32`.

Understanding:

- `get_inputs()` and `get_outputs()` inspect metadata; `session.run()` processes an image.
- The 25,200 rows are candidates (possible predictions)
- Printing the highest-objectness candidate is useful for inspection, but does not
  replace filtering all predictions.

## Day 3 — Decode candidates and restore corner coordinates

Completed:

- Created `detection/decode_frame.py`, reusing `prepare_frame()` from `infer_frame.py`.
- Converted objectness logits to confidence using sigmoid and retained confidence at least `0.5`.
- Selected colour and pattern labels by their highest logits, using the assignment's tables.
- Undid letterbox padding and resizing to recover original-image corner coordinates.
- Drew candidate outlines, corner dots and labels on a copy of the frame.

Verified:

- The first frame of `small.mp4` produces 19 candidates after confidence filtering.
- The highest-objectness candidate is classified as `red`, pattern `3`, confidence about `0.9539`.
- Its first corner maps from about `(318.86, 369.92)` in the model image to
  `(717.44, 652.33)` in the original frame.
- Low-confidence-only input returns no candidates; extreme logits do not overflow sigmoid.
- The user displayed the annotated frame and reviewed the overlapping outlines.

Understanding:

- Confidence is sigmoid(objectness), without multiplying by colour or pattern scores.
- Colour and pattern logits are comparison scores, not percentages.
- The corners represent the two lightbars' endpoints, ordered upper-left, lower-left,
  lower-right, upper-right relative to the plate's orientation.
- Multiple candidates can describe the same plate and produce overlapping outlines.

## Day 4 — NMS and sample-frame checks

Completed:

- Created `detection/nms_frame.py` to remove overlapping duplicate predictions.
- Calculated IoU using the axis-aligned rectangles enclosing the four corners.
- Sorted candidates by confidence, kept the strongest, and suppressed candidates with
  IoU greater than `0.5`, repeating until no candidates remain.
- Applied NMS across all labels so overlapping contradictory labels compete for one prediction.
- Kept the winning candidate's corners, label and confidence unchanged.
- Added `--video`, `--frame-index` and `--no-display` options.

Verified with display disabled:

| Video | Frame index | After confidence filtering | After NMS |
| --- | ---: | ---: | ---: |
| `small.mp4` | 0 | 19 | 2 |
| `large.mp4` | 0 | 28 | 2 |
| `small.mp4` | 180 | 17 | 1 |
| `large.mp4` | 180 | 12 | 1 |

- The first small-video frame retains `red 3` at about `0.9539` and `red 4` at about `0.8658`.
- Checked identical boxes, disjoint boxes and a partial-overlap case with known IoU.
- Checked empty input, highest-confidence duplicate selection, preservation of distinct
  plate locations, and that the input detection list is not modified.
- These sample checks do not establish full-video accuracy.

Understanding:

- NMS selects predictions; it does not average or enlarge corners.
- The enclosing rectangle is only for overlap calculation. The original quadrilateral
  is still used for drawing and will be used for PnP.
- Each run currently inspects one selected frame.

## Day 5 — Camera calibration and PnP pose estimation

Completed:

- Added `detection/inspect_calibration.py` to load `calibration.yaml`.
- Inspected focal lengths, principal point and distortion coefficients; confirmed that
  calibration describes the original 1440 x 1080 images.
- Implemented pose estimation in the module now named `detection/pose_framePnP.py`.
- Selected physical dimensions from the predicted pattern: `1` and `base_big` use
  0.230 x 0.056 m; the other defined patterns use 0.135 x 0.056 m.
- Used `SOLVEPNP_IPPE` with restored original-image corners, preserving corner order.
- Compared candidate poses, excluding non-finite solutions and corners behind the camera,
  then selected the pose with the lowest reprojection RMS error.
- Transformed each physical corner using rotation and translation to obtain camera-relative
  3D coordinates in metres: +X right, +Y down, +Z forward.

Verified and understood:

- The translation vector describes the plate centre; each corner needs its own `R * P + t`.
- Reprojection RMS is `sqrt(sum(squared Euclidean corner errors) / 4)` in original pixels.
- The default 5-pixel quality cutoff rejects questionable poses. A rejected pose retains
  its 2D detection with both 3D corners and reprojection error set to `null`.
- A small reprojection error measures agreement with detected corners; it does not prove
  correct classification, physical dimensions or real-world distance accuracy.

## Day 6 — Full-video processing and JSONL export

Completed:

- Added the final executable, now named `detection/run_video_detection.py`.
- Reused preprocessing, decoding, NMS, calibration and PnP helpers across all frames.
- Required `--model`, `--video`, `--calibration` and `--output` arguments so paths are
  supplied by the caller. One invocation processes one video; run twice for both clips.
- Exported one JSON object per frame, including empty detections and retained failed poses.
- Used `frame_index / 30` timestamps, starting at zero separately for each clip.
- Wrote results through a temporary file, replacing the destination only after successful
  processing. Reruns overwrite by default; `--no-overwrite` protects an existing result.
- Moved the standalone model, video and preprocessing checks into
  `detection/validation_debug/`. Earlier entries record their original locations.
- Kept shared modules in `detection/`; importing them does not run their demonstration code.

Verified from the generated JSONL files:

| Video | Frame records | Detections across all frames | Empty frames | Rejected poses retained |
| --- | ---: | ---: | ---: | ---: |
| `small.mp4` | 360 | 395 | 162 | 0 |
| `large.mp4` | 360 | 150 | 260 | 3 |

- Checked frame order, timestamps, required fields, valid labels, finite coordinates,
  positive depths for exported poses and physical distances between the 3D corners.
- Both files have frame indices 0 through 359. JSON output does not allow NaN or Infinity.
- Detection totals count predictions across frames, not unique tracked plates.

## Day 7 — Documentation and final review (in progress)

Completed:

- Added `IMPLEMENTATION.md` as a separate guide, preserving the original assignment READMEs.
- Documented repository structure, each script's role, the processing pipeline, environment
  setup, terminal/PyCharm run instructions, output format and verification limits.
- Ordered the guide as repository structure, script descriptions, pipeline and full run
  instructions, followed by output and checks.
- Compared local results with another implementation on 2026-11-03, as recorded below.

## Remaining work

- Reproduce the documented workflow in a fresh environment using `requirements.txt`.
- Review more representative frames from both clips for missed plates, duplicate detections,
  corner placement, unstable labels and questionable geometry.
- Review and document any threshold changes; the current confidence/NMS thresholds are
  0.5 and the pose quality cutoff is 5 pixels, without ground-truth optimization.
- Finish submission review against the original requirements. Day 7 remains in progress.
- A direct detection-by-detection comparison would require the other author's JSONL files.

## Git checkpoints

- `2bcd17b`: Python environment setup and model inspection.
- `3c51147`: moved the development log to the repository root.
- `6cf19dd`: single-frame armour detection and NMS.
- `3873eb8`: restored the original assignment README.
- `de987b4`: camera calibration loading and inspection.
- `318b5cc`: PnP pose estimation and camera-relative 3D corners.
- `3b9f230`: full-video JSON export and separation of validation scripts.
- `de20e37`: command-line input and output paths.
- `da7bef6`: renamed the final executable.
- `4794650`: implementation structure and run guide.