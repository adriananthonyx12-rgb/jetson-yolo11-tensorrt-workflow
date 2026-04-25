# Troubleshooting Notes

## Problem: `FileNotFoundError: yolo11n.engine does not exist`

Cause: The TensorRT engine file has not been created or is not in the current folder.

Fix:

```bash
find ~ -name "*.engine"
```

Then update the model path in the code:

```python
model = YOLO("/home/humber/yolo11n.engine", task="detect")
```

## Problem: Wrong PyTorch Version

Error example:

```text
The NVIDIA driver on your system is too old
torch.cuda.is_available(): False
```

Cause: pip installed a generic CUDA PyTorch version that does not match Jetson.

Fix:
- uninstall bad torch
- install Jetson-compatible torch
- avoid reinstalling Ultralytics with dependencies after the correct torch is installed

```bash
pip uninstall -y torch torchvision torchaudio
```

Then reinstall the Jetson-compatible wheel.

## Problem: `No module named onnx`

Cause: ONNX is required during TensorRT export.

Fix:

```bash
pip install onnx onnxslim
```

## Problem: `No module named tensorrt`

Cause: The venv cannot see the TensorRT Python bindings or the bindings are not installed.

Fix:

```bash
sudo apt install -y python3-libnvinfer*
```

Then use a venv with system site packages:

```bash
python3 -m venv --system-site-packages ~/yoloenv_trt
```

## Problem: `ImportError: cannot import name equal_valued from sympy`

Cause: Python is using an old system SymPy version.

Fix:

```bash
pip install "sympy>=1.13"
unset PYTHONPATH
```

## Problem: TensorRT engine input size mismatch

Error example:

```text
input size torch.Size([1, 3, 640, 640]) not equal to max model size (1, 3, 320, 320)
```

Cause: The engine was exported with `imgsz=320`, but the code used `imgsz=640`.

Fix:

```python
results = model.predict(frame, device="cuda", imgsz=320, verbose=False, task="detect")
```

## Problem: TensorRT build takes a long time

This is normal on the Jetson Orin Nano. TensorRT benchmarks kernels and builds an optimized engine for the GPU.

Useful commands:

```bash
tegrastats
jtop
```

Optional performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Problem: OpenCV font warnings

Warning example:

```text
QFontDatabase: Cannot find font directory
```

This warning usually does not stop inference. If the camera window opens and detections work, it can be ignored.
