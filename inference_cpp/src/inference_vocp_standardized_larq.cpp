#include <cstdio>
#include <ctime>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <cmath>
#include <string>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace tflite;
using namespace std;
using namespace cv;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::vector<uint8_t> convert_mat(cv::Mat input, int height, int width,int channels)
{
    std::vector<uint8_t> output(height * width * channels);

    for(int i = 0; i < height; i++){
        int des_pos;
        for(int j = 0; j < width; j++){
        des_pos = (i * width + j) * channels;

        cv::Vec3b intensity = input.at<cv::Vec3b>(i, j);
        output[des_pos] = (uint8_t) intensity.val[2]; //R
        output[des_pos+1] = (uint8_t)intensity.val[1]; //G
        output[des_pos+2] = (uint8_t) intensity.val[0]; //B
        }
    }

    return output;
}

void ProcessInputWithFloatModel(uint8_t* input, float* buffer) {
    const int wanted_input_height = 300;
    const int wanted_input_width = 300;
    const int wanted_input_channels = 300;
    for (int y = 0; y < wanted_input_height; ++y) {
        float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
        for (int x = 0; x < wanted_input_width; ++x) {
            uint8_t* input_pixel = input + (y * wanted_input_width * wanted_input_channels) + (x * wanted_input_channels);
            float* out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                out_pixel[c] = input_pixel[c] / 255.0f;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 1) {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }
    const char* filename = "../../models_trained/ssd_mobilenet_larq.tflite";

    cv::Mat image;
    cv::Mat resized;
    int image_height = 300;
    int image_width = 300;
    int image_channels = 3;
    clock_t time_req, time_req_1, time_req_2;

    //   string img_directory = "~/datasets/VOCdevkit_test/VOC2007/JPEGImages";
    //   string img_csv = "~/ssd-keras/dataset_voc_csv/2007_person_test.csv"
    string img_path = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/000043.jpg";

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // register LCE custom ops
    compute_engine::tflite::RegisterLCECustomOps(&resolver);

    // Build the interpreter
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    time_req_1 = clock();

    // Fill 'input'.
    float* input = interpreter->typed_input_tensor<float>(0);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // printf("=== Pre-invoke Interpreter State ===\n");
    // tflite::PrintInterpreterState(interpreter.get());

    interpreter->SetNumThreads(4);

    // Fill input buffers
    // resize image and load into input  
    image = imread(img_path, cv::IMREAD_COLOR);
    
    if(image.empty())
    {
        std::cout << "Could not read the image: " << img_path << std::endl;
        return 1;
    }

    cv::resize(image, resized, cv::Size(image_width,image_height));
    // std::vector<uint8_t> converted_input = convert_mat(resized,image_height,image_width,image_channels);
    // uint8_t* in = image.ptr<uint8_t>(0);
    // ProcessInputWithFloatModel(in, input);
    
   

    memcpy(interpreter->typed_input_tensor<float>(0), resized.data, resized.total() * resized.elemSize());
	// interpreter->SetAllowFp16PrecisionForFp32(true);
    
    time_req_2 = clock();

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    // printf("\n\n=== Post-invoke Interpreter State ===\n");
    // tflite::PrintInterpreterState(interpreter.get());
    printf("Done invoking.\n");

    float* output = interpreter->typed_output_tensor<float>(0);

    time_req_1 = clock() - time_req_1;
    time_req_2 = clock() - time_req_2;

    cout << "Loading image and inference of model took " << (float)time_req_1/CLOCKS_PER_SEC << " seconds, or equivalent FPS of " << CLOCKS_PER_SEC/(float)time_req_1 << endl;
    cout << "Only inference of model took " << (float)time_req_2/CLOCKS_PER_SEC << " seconds, or equivalent FPS of " << CLOCKS_PER_SEC/(float)time_req_2 << endl;

    // Read output buffers
    // TODO(user): Insert getting data out code.

    return 0;
}