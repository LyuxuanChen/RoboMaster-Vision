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
