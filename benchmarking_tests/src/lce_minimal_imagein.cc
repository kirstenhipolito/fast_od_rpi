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
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void fill_buffer_with_mat(cv::Mat input, float* to_inp, int height, int width,int channels)
{
  //UNOPTIMIZED
  //To do: Find a memcpy version of this
  for(int i = 0; i < height; i++){
    int des_pos;
    for(int j = 0; j < width; j++){
      des_pos = (i * width + j) * channels;
      cv::Vec3b intensity = input.at<cv::Vec3b>(i, j);
      to_inp[des_pos] = intensity.val[2] / 255.0f; //R
      to_inp[des_pos+1] = intensity.val[1] / 255.0f; //G
      to_inp[des_pos+2] = intensity.val[0] / 255.0f; //B
    }
  }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "lce_minimal <tflite model> <num_threads>\n");
        return 1;
    }

    int num_runs = 50;
    float ave_invoke_ms = 0;
    float ave_inference_ms = 0;
    int num_threads = std::atoi(argv[2]);

    const char* filename = argv[1];

    cv::Mat image;
    cv::Mat resized;
    int image_height = 300;
    int image_width = 300;
    int image_channels = 3;

    clock_t time_req_1, time_req_2;
    string img_path = "../datasets/VOCdevkit_test/VOC2007/JPEGImages/000043.jpg";

    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    

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

    for (int i = 0; i < num_runs; i++) {

        time_req_1 = clock(); //time_req_1 -> ave_inference_ms

        
        
        time_req_2 = clock(); //time_req_1 -> ave_invoke_ms

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        float* output = interpreter->typed_output_tensor<float>(0);

        time_req_1 = clock() - time_req_1;
        time_req_2 = clock() - time_req_2;

        std::cout << row[0] << "  |  " << (float)time_req_1*1000/CLOCKS_PER_SEC << "  |  " << (float)time_req_2*1000/CLOCKS_PER_SEC << "  |  " << CLOCKS_PER_SEC/(float)time_req_1 << "  |  " << CLOCKS_PER_SEC/(float)time_req_2 << std::endl;
        
        // Read output buffers
        // TODO(user): Insert getting data out code.
        
        ave_inference_ms += time_req_1;
        ave_invoke_ms += time_req_2;


    }

    std::cout << "Testing invoke() on " << argv[2] << "threads for " << num_runs << "times." << std::endl;

    // Run multiple iterations of invoke
    for (int i = 0; i < num_runs; i++) {
        // Start inference clock
        auto start1 = std::chrono::steady_clock::now();

        // Fill input buffers, resize image and load into input  
        image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "Could not read the image: " << img_path << std::endl;
        } 
        cv::resize(image, resized, cv::Size(image_width,image_height));
        fill_buffer_with_mat(resized,interpreter->typed_input_tensor<float>(0),image_height,image_width,image_channels);

        auto start2 = std::chrono::steady_clock::now();

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        float* output = interpreter->typed_output_tensor<float>(0);

        // Get end clock
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time of invoke (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start2).count() << std::endl;
        std::cout << "Time of inference (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start1).count() << std::endl;
        ave_invoke_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start2).count();
        ave_inference_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start1).count();

    }

    std::cout << "Average invoke time (ms): " << (float)ave_invoke_ms/num_runs << std::endl;
    std::cout << "Average inference time (ms): " << (float)ave_inference_ms/num_runs << std::endl;
    
    return 0;
}
