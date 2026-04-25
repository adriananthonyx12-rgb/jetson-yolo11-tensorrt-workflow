# Jetson Setup Notes

This file documents the setup fixes needed to run YOLO11 with TensorRT on the NVIDIA Jetson Orin Nano.

## 1. Create a Python Environment

```bash
python3 -m venv --system-site-packages ~/yoloenv_trt
source ~/yoloenv_trt/bin/activate
python3 -m pip install --upgrade pip
```

The `--system-site-packages` option is important because TensorRT is installed through JetPack system packages.

## 2. Install Ultralytics and ONNX Tools

Use `--no-deps` if you already installed the correct Jetson PyTorch version and do not want pip to replace it.

```bash
pip install --no-deps ultralytics onnx onnxslim
pip install "sympy>=1.13"
```

## 3. PyTorch Fix for Jetson

A normal pip install may install the wrong PyTorch version, such as a CUDA 13 build. That can cause:

```text
torch.cuda.is_available(): False
```

The fix is to use a Jetson-compatible PyTorch wheel.

Example used in this project:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
```

Verify CUDA:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"
```

Expected result:

```text
torch.cuda.is_available() = True
```

## 4. TensorRT Python Binding Fix

TensorRT should be installed through JetPack packages, not pip.

Check packages:

```bash
dpkg -l | grep -E 'nvinfer|tensorrt'
```

Install if needed:

```bash
sudo apt update
sudo apt install -y python3-libnvinfer*
```

Verify:

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## 5. Export YOLO11 to TensorRT

Use 320 x 320 first because it is easier to build on the Jetson Orin Nano.

```bash
yolo export model=yolo11n.pt format=engine imgsz=320 half=True
```

This creates:

```text
yolo11n.engine
```

## 6. Run GPU TensorRT Inference

```bash
python3 Workflow/code/run_yolo.py
```

## 7. Run CPU Inference

```bash
python3 Workflow/code/run_yolocpu.py
```

## 8. Monitor Jetson Performance

Install jetson-stats:

```bash
sudo -H pip3 install -U jetson-stats
sudo reboot
```

Run:

```bash
jtop
```

Or use:

```bash
tegrastats
```

Important metrics:
- GPU usage: GR3D
- RAM usage
- CPU usage
- temperature
- swap usage
