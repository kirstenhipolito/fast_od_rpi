#include <cstdio>
#include <ctime>
#include <iostream>
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
  if (argc != 2) {
    fprintf(stderr, "lce_minimal <tflite model>\n");
    return 1;
  }

  int num_runs = 50;
  clock_t ave_invoke_ms = 0;
  clock_t time_req_1;

  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  
  // Initial invoke
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Run multiple iterations of invoke
  for (int i = 0; i < num_runs; i++) {    
      time_req_1 = clock();
      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      time_req_1 = clock() - time_req_1;

      std::cout << "Time of invoke (s/FPS): " << (float)time_req_1/CLOCKS_PER_SEC << " / " << CLOCKS_PER_SEC/(float)time_req_1 << std::endl;
      ave_invoke_ms += time_req_1;
  }

  std::cout << "Average invoke time (ms): " << (float)ave_invoke_ms*1000/(CLOCKS_PER_SEC*num_runs) << std::endl;

  return 0;
}
