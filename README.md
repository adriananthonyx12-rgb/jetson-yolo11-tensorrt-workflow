# YOLO11 Object Detection on NVIDIA Jetson Orin Nano

## Project Overview

This project demonstrates real-time object detection on an embedded GPU edge device using an NVIDIA Jetson Orin Nano, YOLO11, CUDA, TensorRT, and OpenCV. The main goal is to compare object detection inference using CPU execution and GPU-accelerated TensorRT execution.

The workflow uses a live camera feed, runs object detection, and displays annotated frames with bounding boxes and class labels.

## Problem Statement

Object detection is a parallel computation problem because a neural network performs many matrix operations, convolution operations, and tensor calculations at the same time. These operations are much faster when they are accelerated by a GPU instead of only using a CPU.

For this project, the Jetson Orin Nano is used because it provides an embedded platform with CUDA-capable GPU acceleration. YOLO11n was selected because it is a lightweight object detection model that is suitable for real-time embedded applications.

## Selected Hardware and Software

### Hardware
- NVIDIA Jetson Orin Nano
- USB camera or CSI camera
- Monitor, keyboard, and mouse
- MicroSD or NVMe storage

### Software
- Ubuntu through NVIDIA JetPack
- Python 3.10
- Ultralytics YOLO
- PyTorch for Jetson
- TensorRT
- ONNX and ONNXSlim
- OpenCV
- jetson-stats / jtop
- tegrastats

### Model
- YOLO11n pretrained model

YOLO11n was selected because it is small, fast, and appropriate for real-time inference on an embedded GPU platform.

## Repository Structure

```text
jetson-yolo11-tensorrt-workflow/
├── README.md
├── setup.md
├── troubleshooting.md
├── Workflow/
│   ├── README.md
│   └── code/
│       ├── run_yolo.py
│       └── run_yolocpu.py
├── Reflection-Learning-Plan/
│   └── README.md
└── assets/
```

## Quick Start

Activate the environment:

```bash
source ~/yoloenv_trt/bin/activate
```

Run TensorRT GPU inference:

```bash
python3 Workflow/code/run_yolo.py
```

Run CPU inference:

```bash
python3 Workflow/code/run_yolocpu.py
```

## Main Result

The TensorRT engine version uses GPU acceleration and is much faster than the CPU version. The CPU version is mainly useful for testing and comparison, while the TensorRT engine is the optimized version for real-time edge inference.

## Notes

The TensorRT engine was exported using:

```bash
yolo export model=yolo11n.pt format=engine imgsz=640 half=True
```

The inference code must use the same image size:

```python
imgsz=640
```

If the engine was built at 320 x 320, running it at 640 x 640 will cause an input-size mismatch error.
