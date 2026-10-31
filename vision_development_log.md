# Vision development log

Selected route: armour detection and pose estimation, implemented in Python.

Updated: 2026-10-31. Day numbers refer to the seven planned learning milestones,
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

## Remaining milestones

- Continue reviewing representative frames from both clips and document threshold changes.
- Day 5: load calibration, select physical dimensions from the predicted pattern, estimate
  pose with PnP, recover 3D corners and check reprojection error.
- Day 6: process all frames of both videos and export the required JSONL, including empty
  detections and failed-pose handling.
- Day 7: verify the complete workflow from a fresh environment and finish submission documentation.

## Git checkpoints

- Day 1 setup and model inspection were committed and pushed as `2bcd17b`.
- The development log was moved to the repository root in `3c51147`.
- At the time of this entry, later scripts and this documentation update await review and commit.
- Keep commits for video/inference, decoding and NMS separate where practical to show the
  development sequence. Review selected files in PyCharm before committing.
