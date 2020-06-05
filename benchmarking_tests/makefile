COMPILATION FOR BENCH TESTS:

** this assumes a directory layout of the following:

(working directory):
|-- tensorflow
|-- compute-engine
|-- datasets
|   |-- VOCdevkit_test
|   +-- 2007_person_test.csv (get it here: https://github.com/henritomas/ssd-keras/blob/master/dataset_voc_csv/2007_person_test.csv)
+-- fast_od_rpi
    |-- inference_cpp
    |-- benchmarking_tests
    +-- models_trained

** compile within benchmarking_tests

** replace <arch> with : "linux_aarch64" if using macOS
                         "linux_aarch64" if using RPi 4
                         linux_aarch64

TFLITE BENCHMARK TOOL:
g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/tflite_benchmark_main.cc  -o fast_od_rpi/benchmarking_tests/obj/tflite_benchmark_main.o

g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/usr/local/include \
-o fast_od_rpi/benchmarking_tests/bin/tflite_benchmark_main fast_od_rpi/benchmarking_tests/obj/tflite_benchmark_main.o \
tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/benchmark-lib.a -lstdc++ -lpthread -lm -lz -ldl


LARQ BENCHMARK TOOL:
g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -Icompute-engine -Icompute-engine/larq_compute_engine/tflite/cc -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/lce_benchmark_main.cc  -o fast_od_rpi/benchmarking_tests/obj/lce_benchmark_main.o

g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -Icompute-engine -Ilarq_compute_engine/tflite/cc -I. -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/usr/local/include \
-o fast_od_rpi/benchmarking_tests/bin/lce_benchmark_main fast_od_rpi/benchmarking_tests/obj/lce_benchmark_main.o \
compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a compute-engine/third_party/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/benchmark-lib.a -lstdc++ -lpthread -lm -lz -ldl

MINIMAL_NOINPUT:
g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK  -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/minimal_noinput.cc -o fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o

g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -I. -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include \
	-o fast_od_rpi/benchmarking_tests/bin/minimal_noinput fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o \
	 tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a  -lstdc++ -lpthread -lm -lz -ldl

     

MINIMAL_NOINPUT MANJARO:

aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC  -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/minimal_noinput.cc -o fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o

aarch64-linux-gnu-g++ -O3 -DNDEBUG -fPIC -DTFLITE_WITH_RUY --std=c++11 -march=armv8-a -funsafe-math-optimizations -ftree-vectorize -fPIC  -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include \
	-o fast_od_rpi/benchmarking_tests/bin/minimal_noinput fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o \
	 tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a  -Wl,--no-export-dynamic -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--as-needed -lrt -lstdc++ -lpthread -lm -ldl

MINIMAL_IMAGEIN:
g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK  -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/minimal_imagein.cc -o fast_od_rpi/benchmarking_tests/obj/minimal_imagein.o `pkg-config --cflags --libs opencv`

g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -I. -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include \
	-o fast_od_rpi/benchmarking_tests/bin/minimal_imagein fast_od_rpi/benchmarking_tests/obj/minimal_imagein.o \
	 tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a  -lstdc++ -lpthread -lm -lz -ldl `pkg-config --cflags --libs opencv`

INFERENCE


