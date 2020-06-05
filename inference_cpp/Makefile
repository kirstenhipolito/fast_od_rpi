inference_camera:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../tensorflow -I../../tensorflow/tensorflow/lite/tools/make/downloads/ -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../tensorflow/tensorflow/lite/tools/make/downloads/absl -I../../tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -I../../tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I./lib -I/usr/local/include -c src/inference_camera.cpp -o obj/inference_camera.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -c lib/decode_detections.cpp -o obj/decode_detections.o

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../tensorflow -I../../tensorflow/tensorflow/lite/tools/make/downloads/ -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../tensorflow/tensorflow/lite/tools/make/downloads/absl -I../../tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -I../../tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I./lib -I/usr/local/include \
-o bin/inference_camera obj/inference_camera.o obj/decode_detections.o \
 ../../tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a -Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

decode_detections:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC -I../../tensorflow/tensorflow/lite/tools/make/downloads/eigen -c lib/decode_detections.cpp -o obj/decode_detections.o

clean:
	rm -rf bin
	mkdir bin
