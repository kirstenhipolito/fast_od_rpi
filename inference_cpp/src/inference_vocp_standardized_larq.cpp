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
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>

#include "lib/decode_detections.hpp"
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

class CSVRow
{
    public:
        std::string const& operator[](std::size_t index) const
        {
            return m_data[index];
        }
        std::size_t size() const
        {
            return m_data.size();
        }
        void readNextRow(std::istream& str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
            std::string         cell;

            m_data.clear();
            while(std::getline(lineStream, cell, ','))
            {
                m_data.push_back(cell);
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())
            {
                // If there was a trailing comma then add an empty element.
                m_data.push_back("");
            }
        }
    private:
        std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
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

    string img_directory = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/";
    string img_csv = "../../ssd-keras/dataset_voc_csv/2007_person_test.csv";
    string img_path = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/000043.jpg";

    std::ifstream file(img_csv);
    CSVRow row;
    while(file >> row)
    {
        full_path = img_directory + row[0];
        std::cout << full_path << std::endl;
    }

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

    interpreter->SetNumThreads(4);
    // interpreter->SetAllowFp16PrecisionForFp32(true);

    time_req_1 = clock();

    // Fill 'input'.
    float* input = interpreter->typed_input_tensor<float>(0);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // printf("=== Pre-invoke Interpreter State ===\n");
    // tflite::PrintInterpreterState(interpreter.get());

    // Fill input buffers
    // resize image and load into input  
    image = imread(img_path, cv::IMREAD_COLOR);
    
    if(image.empty())
    {
        std::cout << "Could not read the image: " << img_path << std::endl;
        return 1;
    }

    cv::resize(image, resized, cv::Size(image_width,image_height));
    memcpy(interpreter->typed_input_tensor<float>(0), resized.data, resized.total() * resized.elemSize());
	
    
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
