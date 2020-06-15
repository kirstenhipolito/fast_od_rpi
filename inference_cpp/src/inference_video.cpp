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
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "decode_detections.hpp"

#include <Eigen/Dense>

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

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "./bin/inference_video <(string) tflite model> <(float) confidence_threshold> <(float) iou_threshold> <(int) od_mode: 1=multi, 2=person> <(string) video file>\n");
        return 1;
    }

    char* filename = argv[1];
    
    int num_runs = 0;
    float ave_loadtime_ms = 0;
    float ave_invoke_ms = 0;
    float ave_predec_ms = 0;
    float ave_decoding_ms = 0;
    float ave_inference_ms = 0;
    float ave_display_ms = 0;
    float ave_whole_ms = 0;
    int num_threads = 4;

    int od_mode = (int) std::stoi(argv[4]);

    const int y_pred_rows = 2268;
  	int y_pred_cols = 0;

    switch (od_mode) {
        case 1:
            y_pred_cols = 33;
            break;
        case 2:
            y_pred_cols = 14;
            break;
    }

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

    char* videoname = argv[5];

    cv::VideoCapture video(videoname); //capture the video from rpi camera

    if (!video.isOpened() )  // if not success, exit program
    {
         std::cout << "Cannot open the video file" << std::endl;
         return -1;
    }

    int frame_width = video.get(cv::CAP_PROP_FRAME_WIDTH); 
    int frame_height = video.get(cv::CAP_PROP_FRAME_HEIGHT); 

    cv::VideoWriter video_write("out_vid.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, Size(frame_width,frame_height));

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

    std::cout << "Press any key to end." << "\n";
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    // cv::namedWindow("Video View", cv::WINDOW_AUTOSIZE);

    while(1)
    {
        auto start_inference = std::chrono::steady_clock::now();
        // Load image into input
        video >> image;

        if (image.empty())
            break;

        cv::resize(image, resized, cv::Size(image_width,image_height));

        fill_buffer_with_mat(resized,interpreter->typed_input_tensor<float>(0),image_height,image_width,image_channels);

        auto start_invoke = std::chrono::steady_clock::now();

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

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

        auto start_decoding = std::chrono::steady_clock::now();

        vec_boxes = decode_detections((Eigen::MatrixXf) y_pred, confidence_thresh, iou_thresh, top_k, image_height, image_width);

        auto end_inference = std::chrono::steady_clock::now();

        draw_bounding_boxes_custom_dim(image,vec_boxes,od_mode,frame_width,frame_height);
        // cv::imshow("Video View",resized);

        video_write.write(image);

        auto end_livestream = std::chrono::steady_clock::now();

        std::cout << "Current total ms/FPS: " << (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_livestream - start_inference).count() << " / " << 1000/(float)(std::chrono::duration_cast<std::chrono::milliseconds>(end_livestream - start_inference).count()) << std::endl;
        ave_loadtime_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(start_invoke - start_inference).count();
        ave_invoke_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke).count();
        ave_predec_ms += (float) std::chrono::duration_cast<std::chrono::microseconds>(start_decoding - end_invoke).count();
        ave_decoding_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_decoding).count();
        ave_inference_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count();
        ave_display_ms += (float) std::chrono::duration_cast<std::chrono::microseconds>(end_livestream - end_inference).count();
        ave_whole_ms += (float) std::chrono::duration_cast<std::chrono::milliseconds>(end_livestream - start_inference).count();
        num_runs += 1;

      if(cv::waitKey(30) >= 0) break;
    }

    std::cout << "Average image load time (ms/FPS): " << (float)ave_loadtime_ms/num_runs << " / " << num_runs/((float)ave_loadtime_ms/1000) << std::endl;
    std::cout << "Average invoke time (ms/FPS): " << (float)ave_invoke_ms/num_runs << " / " << num_runs/((float)ave_invoke_ms/1000) << std::endl;
    std::cout << "Average pre-decoding time (ms/FPS): " << (float)(ave_predec_ms/1000)/num_runs << " / " << num_runs/((float)(ave_predec_ms/1000)/1000) << std::endl;
    std::cout << "Average decoding time (ms/FPS): " << (float)ave_decoding_ms/num_runs << " / " << num_runs/((float)ave_decoding_ms/1000) << std::endl;
    std::cout << "Average display time (ms/FPS): " << (float)(ave_display_ms/1000)/num_runs << " / " << num_runs/((float)(ave_display_ms/1000)/1000) << std::endl << std::endl;

    std::cout << "Average inference time (ms/FPS): " << (float)ave_inference_ms/num_runs << " / " << num_runs/((float)ave_inference_ms/1000) << std::endl;
    std::cout << "Average livestream time (ms/FPS): " << (float)ave_whole_ms/num_runs << " / " << num_runs/((float)ave_whole_ms/1000) << std::endl;

    video.release();
    video_write.release();


    cv::destroyAllWindows();


    return 0;
}
