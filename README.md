# Faster Object Detection on Raspberry Pi

This project performs faster object detection on the Raspberry Pi 4 B through the use of binarized models built with Larq. The submodules included within are:

- `benchmarking_tests`: Benchmarking tests to measure the inference speed of TFLite models.
- `inference_cpp`: The code for performing actual inference for object detection. This is the end-to-end pipeline.
- `decoder`: C++ implementations of NMS and decode_detections built for RPi.
- `models_trained`: TFLite files (.tflite files) of various models. Some are binarized models.

**How to compile for 64-bit Manjaro on a Raspberry Pi 4:**

Within inference_cpp, run
```aarch64-unknown-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=native -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../tensorflow -I../../tensorflow/tensorflow/lite/tools/make/downloads/ -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../tensorflow/tensorflow/lite/tools/make/downloads/absl -I../../tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -I../../tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I./lib -I/usr/local/include -c src/inference_camera.cpp -o obj/inference_trial.o `pkg-config --cflags --libs opencv4`
```

```
aarch64-unknown-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=native -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../tensorflow -I../../tensorflow/tensorflow/lite/tools/make/downloads/ -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../tensorflow/tensorflow/lite/tools/make/downloads/absl -I../../tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -I../../tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I./lib -I/usr/local/include \
-o bin/inference_trial obj/inference_trial.o obj/decode_detections.o \
 ../../tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a -Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`
 ```
 Run within inference_cpp with:
 `./bin/inference_trial`
