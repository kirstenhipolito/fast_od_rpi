#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>

#include "decode_detections.hpp"

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
using namespace std;
using namespace cv;

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
    if (argc != 2) {
        fprintf(stderr, "./bin/inference_trial <tflite model>\n");
        return 1;
    }

    char* filename = argv[1];

    int num_runs = 50;
    float ave_invoke_ms = 0;
    float ave_inference_ms = 0;
    int num_threads = 4;

    cv::Mat image;
    cv::Mat resized;
    int image_height = 300;
    int image_width = 300;
    int image_channels = 3;
    float confidence_thresh = 0.3;
    float iou_thresh = 0.45;
    int top_k = 4;
    
    string img_directory = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/";
    string img_csv = "../../datasets/2007_person_test.csv";
    // string img_path = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/000043.jpg";
    string img_path = "";

    std::ifstream file(img_csv);
    CSVRow row;

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // register LCE custom ops
    compute_engine::tflite::RegisterLCECustomOps(&resolver);

    // Build the interpreter
    std::unique_ptr<Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter,num_threads);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Fill 'input'.
    float* to_inp = interpreter->typed_input_tensor<float>(0);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    file >> row;

    std::cout << std::fixed;
    std::cout << std::setprecision(6);

    for (int i = 0; i < num_runs; i++) {
        file >> row;
        img_path = img_directory + row[0];

        auto start_inference = std::chrono::steady_clock::now();

        // Fill input buffers, resize image and load into input  
        image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "Could not read the image: " << img_path << std::endl;
        } 
        cv::resize(image, resized, cv::Size(image_width,image_height));
        // memcpy(interpreter->typed_input_tensor<float>(0), image.data, image.total() * image.elemSize());
        fill_buffer_with_mat(resized,interpreter->typed_input_tensor<float>(0),image_height,image_width,image_channels);
        
        auto start_invoke = std::chrono::steady_clock::now();

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        
        // Get end clock
        auto end_invoke = std::chrono::steady_clock::now();

        float* output = interpreter->typed_output_tensor<float>(0);

        auto end_inference = std::chrono::steady_clock::now();

        std::cout << "Time of invoke (ms/FPS): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke).count() << " / " << 1/(std::chrono::duration_cast<std::chrono::seconds>(end_invoke - start_invoke).count()) << std::endl;
        std::cout << "Time of inference (ms/FPS): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count() << " / " << 1/(std::chrono::duration_cast<std::chrono::seconds>(end_inference - start_inference).count()) << std::endl;
        ave_invoke_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start2).count();
        ave_inference_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end - start1).count();

    }

    std::cout << "Average invoke time (ms/FPS): " << (float)ave_invoke_ms/num_runs << " / " << num_runs/((float)ave_invoke_ms*1000) << std::endl;
    std::cout << "Average inference time (ms/FPS): " << (float)ave_inference_ms/num_runs << " / " << num_runs/((float)ave_invoke_ms*1000) << std::endl;

    return 0;
}
