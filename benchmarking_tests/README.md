Benchmarking Tests

A set of tests lifted from the TFLite and LARQ libraries to benchmark the performance of quantized and binarized models. Binaries available on /benchmarking_tests/bin are compiled for Raspberry Pi 4 on Manjaro 20.02 (64-bit).

Instructions on how to use:
    * Run these with fast_od_rpi as the working directory.
    * To run "benchmark_model" binaries:
    `./benchmarking_tests/bin/[framework]_benchmark_model --graph=models_trained/[tflite_file] --num_threads=[num_threads]`
    * To run "minimal_noinput" binaries:
    `./benchmarking_tests/bin/[framework]_minimal_noinput [tflite_file] [num_threads]`
    * To run "mininal_imagein" binaries:
    `./benchmarking_tests/bin/[framework]_minimal_imagein [tflite_file] [num_threads]`

See compilation_instructions.txt for instructions on compilation.