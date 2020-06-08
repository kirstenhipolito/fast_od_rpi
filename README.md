# Faster Object Detection on Raspberry Pi

This project performs faster object detection on the Raspberry Pi 4 B through the use of binarized models built with Larq. The submodules included within are:

- `benchmarking_tests`: Benchmarking tests to measure the inference speed of TFLite models.
- `inference_cpp`: The code for performing actual inference for object detection. This is the end-to-end pipeline.
- `decoder`: C++ implementations of NMS and decode_detections built for RPi.
- `models_trained`: TFLite files (.tflite files) of various models. Some are binarized models.