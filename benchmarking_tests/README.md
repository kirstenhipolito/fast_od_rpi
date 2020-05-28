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
    +-- benchmarking_tests

** compile within working directory

** replace <arch> with : "osx_x86_64" if using macOS
                         "rpi_armv7l" if using RPi 4

TFLITE BENCHMARK TOOL:



LARQ BENCHMARK TOOL:




MINIMAL_NOINPUT:
g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK  -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include -c fast_od_rpi/benchmarking_tests/src/minimal_noinput.cc -o fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o

g++ -O3 -DNDEBUG -fPIC --std=c++11 -DTFLITE_WITH_RUY -DTFLITE_WITHOUT_XNNPACK -I. -Itensorflow -Itensorflow/tensorflow/lite/tools/make/downloads/ -Itensorflow/tensorflow/lite/tools/make/downloads/eigen -Itensorflow/tensorflow/lite/tools/make/downloads/absl -Itensorflow/tensorflow/lite/tools/make/downloads/gemmlowp -Itensorflow/tensorflow/lite/tools/make/downloads/ruy -Itensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse -Itensorflow/tensorflow/lite/tools/make/downloads/farmhash/src -Itensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include -Itensorflow/tensorflow/lite/tools/make/downloads/fp16/include -I -I/usr/local/include \
	-o fast_od_rpi/benchmarking_tests/obj/minimal_noinput fast_od_rpi/benchmarking_tests/obj/minimal_noinput.o \
	 tensorflow/tensorflow/lite/tools/make/gen/<arch>/lib/libtensorflow-lite.a  -lstdc++ -lpthread -lm -lz -ldl




MINIMAL_IMAGEIN:





INFERENCE