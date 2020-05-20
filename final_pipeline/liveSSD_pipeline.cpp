#include <ctime>
#include <iostream>
#include <cstdio>
#include <raspicam/raspicam_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
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

int main (int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "liveSSD_pipeline <tflite model relative path>\n");
    return 1;
  }

  const char* filename = argv[1];
  time_t timer_begin,timer_end;
	raspicam::RaspiCam_Cv Camera;
	cv::Mat image;
  cv::Mat resized;
  int height = 480;
  int width = 640;
  int channels = 3;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  printf("=== Pre-invoke Interpreter Output State (first 6 rows) ===\n");
  int pre_out_idx = interpreter->outputs()[0];
  float* pre_out = interpreter->typed_tensor<float>(pre_out_idx);
  for(int n = 0; n < (6 * 33); n++){
    std::cout << pre_out[n] << " ";
    if((n%33 == 0) && (n != 0)){
      std::cout << "\n";
    }
  }


  //camera currently set to capture frames of BGR color matrix format
  Camera.set( cv::CAP_PROP_FORMAT, CV_8UC3 );

  std::cout<< "\nOpening Camera..." << "\n";
	if (!Camera.open()) {
    std::cerr<<"Error opening the camera"<< "\n";
    return -1;
  }

  std::cout << "Starting display..."<< "\n";
  std::cout << "Press ESC to begin bounding box prediction." << "\n";

  cv::namedWindow("Camera View", cv::WINDOW_AUTOSIZE);

  //Test video-stream
  while(1){
    Camera.grab();
		Camera.retrieve(image);
    cv::resize(image, resized, cv::Size(width,height));
		cv::imshow("Camera View",resized);
    char c = cv::waitKey(1);
    if(c == 27) break;

  }
  std::cout << "Press ESC to end." << "\n";

  //Proper pipeline
  while(1){
    Camera.grab();
    Camera.retrieve(image);
    cv::resize(image, resized, cv::Size(width,height));

    //Convert Mat resized to std::vector<uin8_t> to make compatible w/ tflite
    std::vector<uint8_t> input = convert_mat(resized,height,width,channels);

    /**
        TFLITE INFERENCE METHOD CALL HERE.
    **/

    //To do: Draw bounding boxes on resized
    cv::imshow("Camera View",resized);

    char c = cv::waitKey(1);
    if(c == 27) break;
  }

  Camera.release();

}
