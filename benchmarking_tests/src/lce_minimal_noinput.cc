#include <cstdio>
#include <ctime>
#include <iostream>
#include <chrono>
#include <string>
#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example where the
// "BuiltinObResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model form disk and perform inference by using the
// C++ interface. See the BUILD file in this directory to see an example of
// linking "Larq Compute Engine" cutoms ops to your inference binary.

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
    compute_engine::tflite::RegisterLCECustomOps(&resolver);

    std::unique_ptr<Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter,num_threads);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // Initial invoke
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
