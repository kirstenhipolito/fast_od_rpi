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
#include <dirent.h>
#include <cstddef>   

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
      to_inp[des_pos] = (float) intensity.val[2]; //R
      to_inp[des_pos+1] = (float) intensity.val[1]; //G
      to_inp[des_pos+2] = (float) intensity.val[0]; //B
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
};

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("If using make, proper format: make run_inference_dataset model=<path to model> conf=<float> iou=<float> data=<float>\n");
        printf("E.g.: make run_inference_dataset model=../models_trained/binarized_ssd_x.tflite conf=0.5 iou=0.3 data=3\n");
        fprintf(stderr, "./bin/inference_dataset <(string) tflite model> <(float) confidence_threshold> <(float) iou_threshold> <(int) dataset_toggle: 1=VOC, 2=person, 3=test_images>\n");
        return 1;
    }

    char* filename = argv[1];

    std::cout << "Using model: " << filename << std::endl;

    int num_runs = 50;
    float ave_invoke_ms = 0;
    float ave_inference_ms = 0;
    int num_threads = 4;

    const int y_pred_rows = 2268;
  	const int y_pred_cols = 33;
    cv::Mat image;
    cv::Mat resized;
    Eigen::MatrixXf y_pred(y_pred_rows, y_pred_cols);
    Eigen::MatrixXf vec_boxes;

    int image_height = 300;
    int image_width = 300;
    int image_channels = 3;
    float confidence_thresh = (float) std::stod(argv[2]);
    float iou_thresh = (float) std::stod(argv[3]);
    int top_k = 200;

    std::cout << "Using confidence threshold: " << confidence_thresh << std::endl;
    std::cout << "Using iou threshold: " << iou_thresh << std::endl;

    int dataset_toggle = (int) std::stoi(argv[4]);

    std::vector<string> img_path_vec;
    string img_path = "";
    string img_save_name = "";

    if (dataset_toggle == 1) { //VOC
        std::cout << "Dataset toggle: VOC2007 FULL test directory." << std::endl;
        string img_dir = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/";
        
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (img_dir.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
                std::string filename(ent->d_name);
                if (filename.find(".jpg") != string::npos){
                    img_path_vec.push_back(img_dir+filename);
                }
                
            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }
        
        num_runs = img_path_vec.size();

    } else if (dataset_toggle == 2) { //person
        std::cout << "Dataset toggle: VOC2007 PERSON test directory." << std::endl;
        string img_dir = "../../datasets/VOCdevkit_test/VOC2007/JPEGImages/";
        string img_csv = "../../datasets/2007_person_test.csv";

        std::ifstream file(img_csv);
        CSVRow row;
        file >> row;

        while (file >> row) {
            img_path_vec.push_back(img_dir + row[0]);
        }

        // num_runs = img_path_vec.size();
        num_runs = 50;

    } else if (dataset_toggle == 3) { //test_images
        std::cout << "Dataset toggle: test_images directory" << std::endl;
        string img_dir = "../../datasets/test_images/";

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (img_dir.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
                std::string filename(ent->d_name);
                if (filename.find(".jpg") != string::npos){
                    img_path_vec.push_back(img_dir+filename);
                }
                
            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }
        
        num_runs = img_path_vec.size();
    }

    std::cout << "Images: " << std::endl;
    for (auto const& i: img_path_vec) {
		std::cout << i << std::endl;
	}

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

    std::cout << std::fixed;
    std::cout << std::setprecision(6);

    for (int i = 0; i < num_runs; i++) {
        img_path = img_path_vec[i];

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

        // Get output, decode, and draw bounding boxes
        int output_idx = interpreter->outputs()[0];
        float* output = interpreter->typed_tensor<float>(output_idx);

        for(int i = 0; i < y_pred_rows; i++){
            int des_pos;
            for (int j = 0; j < y_pred_cols; j++){
                des_pos = (i * y_pred_cols + j);
                y_pred(i,j) = output[des_pos];
            }
        }

        vec_boxes = decode_detections((Eigen::MatrixXf) y_pred, confidence_thresh, iou_thresh, top_k, image_height, image_width);
        

        img_save_name = "out/"+(img_path.substr(img_path.find_last_of("/") + 1)); 

        draw_bounding_boxes_save(resized,vec_boxes, img_save_name);

        auto end_inference = std::chrono::steady_clock::now();
        std::cout << "Image: \n" << img_path << std::endl;
        std::cout << "Vec boxes: \n" << vec_boxes << std::endl;
        std::cout << "Time of invoke (ms/FPS): " << (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke).count() << " / " << 1000/(float)(std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke).count()) << std::endl;
        std::cout << "Time of inference (ms/FPS): " << (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count() << " / " << 1000/(float)(std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count()) << std::endl;
        std::cout << std::endl;
        ave_invoke_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke).count();
        ave_inference_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count();

    }

    std::cout << "Average invoke time (ms/FPS): " << (float)ave_invoke_ms/num_runs << " / " << num_runs/((float)ave_invoke_ms/1000) << std::endl;
    std::cout << "Average inference time (ms/FPS): " << (float)ave_inference_ms/num_runs << " / " << num_runs/((float)ave_inference_ms/1000) << std::endl;

    return 0;
}
