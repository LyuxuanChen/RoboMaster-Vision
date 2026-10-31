# Load the pretrained ONNX model and print its expected input and output
# formats without running detection.

from pathlib import Path

import onnxruntime as ort


# Locate the model relative to this script.
model_path = Path(__file__).resolve().parent / "assets" / "0526.onnx"

if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model using your computer's CPU.
session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

print("Model loaded successfully!")

# Show what data the model expects.
for tensor in session.get_inputs():
    print("INPUT:", tensor.name, tensor.shape, tensor.type)

# Show what data the model produces.
for tensor in session.get_outputs():
    print("OUTPUT:", tensor.name, tensor.shape, tensor.type)