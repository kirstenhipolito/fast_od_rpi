/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "lce_minimal <tflite model> <num_threads>\n");
        return 1;
    }

    int num_runs = 50;
    float ave_invoke_ms = 0;
    int num_threads = std::atoi(argv[2]);

    const char* filename = argv[1];
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter,num_threads);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // Initial invoke()
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    std::cout << "Testing invoke() on " << argv[2] << " threads for " << num_runs << " times." << std::endl;

    // Run multiple iterations of invoke
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::steady_clock::now();
        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time of invoke (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        ave_invoke_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // ave_invoke_ms += time_req_1;
    }

    std::cout << "Average invoke time (ms): " << (float)ave_invoke_ms/num_runs << std::endl;
    return 0;
}
