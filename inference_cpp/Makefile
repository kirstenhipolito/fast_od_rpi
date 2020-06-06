SHELL:=/bin/bash

inference_dataset:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include -c src/inference_dataset.cpp -o obj/inference_dataset.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include \
	-o bin/inference_dataset obj/inference_dataset.o obj/decode_detections.o \
	../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
	-Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

run_inference_dataset: ./bin/inference_camera
	rm -rf out
	mkdir out
	./bin/inference_dataset $(model) $(conf) $(iou) $(data) | tee out/output.txt
	filename=$(basename $(model))
	echo $(filename)
	zip -r out_$(basename $(model))_$(conf)_$(iou)_$(data).zip out
	

inference_camera_unknowngcc:
		aarch64-unknown-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
		-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
		-I./lib -I/usr/local/include -c src/inference_camera.cpp -o obj/inference_camera.o `pkg-config --cflags --libs opencv4`

		aarch64-unknown-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
		-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
		-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
		-I./lib -I/usr/local/include \
		-o bin/inference_camera obj/inference_camera.o obj/decode_detections.o \
		../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
		-Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

inference_camera:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include -c src/inference_camera.cpp -o obj/inference_camera.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include \
	-o bin/inference_camera obj/inference_camera.o obj/decode_detections.o \
	../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
	-Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

decode_detections:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -c lib/decode_detections.cpp -o obj/decode_detections.o `pkg-config --cflags --libs opencv4`

all:
	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -c lib/decode_detections.cpp -o obj/decode_detections.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include -c src/inference_dataset.cpp -o obj/inference_dataset.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include \
	-o bin/inference_dataset obj/inference_dataset.o obj/decode_detections.o \
	../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
	-Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include -c src/inference_camera.cpp -o obj/inference_camera.o `pkg-config --cflags --libs opencv4`

	aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC \
	-I../../compute-engine/larq_compute_engine/tflite/cc -I../../compute-engine -I../../compute-engine/third_party/tensorflow -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/ \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/eigen -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/absl \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
	-I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -I../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I./lib -I/usr/local/include \
	-o bin/inference_camera obj/inference_camera.o obj/decode_detections.o \
	../../compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
	-Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lstdc++ -lpthread -lm -ldl `pkg-config --cflags --libs opencv4`

clean:
	rm -rf bin
	rm -rf obj
	mkdir bin
	mkdir obj

rm-out:
	rm -rf out
	mkdir out
