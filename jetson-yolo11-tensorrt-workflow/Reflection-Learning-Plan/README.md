# Reflection and Learning Plan

## Individual Reflection

During this project, I learned how to set up a YOLO11 object detection workflow on an NVIDIA Jetson Orin Nano. I learned that using GPU acceleration on an embedded platform requires more than just installing normal Python packages. The PyTorch version, CUDA version, TensorRT version, and Python environment must all match the Jetson software stack.

One issue that went well was successfully exporting the YOLO11 model to ONNX and then building a TensorRT engine. This showed that the model could be optimized for GPU inference on the edge device.

One major challenge was fixing package conflicts. The first PyTorch version installed was not compatible with the Jetson driver, so CUDA was not available. Another issue was that TensorRT was installed through JetPack, but the virtual environment could not see the TensorRT Python bindings. This was fixed by creating a virtual environment with system site packages.

I also learned how to compare CPU inference and TensorRT GPU inference. The CPU version is useful for testing, but the TensorRT version is better for real-time object detection.

## Individual Learning Plan

To improve this project further, I would learn more about:

- TensorRT engine optimization
- CUDA memory usage on Jetson devices
- OpenCV camera pipelines
- FPS measurement and benchmarking
- Jetson power modes and thermal management
- Deploying AI models in real embedded systems

For a larger industry project, I would also need more experience with model quantization, INT8 calibration, camera synchronization, deployment scripts, and long-term reliability testing.

Useful future resources:
- NVIDIA Jetson documentation
- Ultralytics YOLO documentation
- TensorRT documentation
- Jetson forums
- OpenCV documentation
