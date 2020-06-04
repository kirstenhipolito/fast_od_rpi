#include <ctime>
#include <iostream>
#include <cstdio>
/*#include <raspicam/raspicam_cv.h>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/core/matx.hpp>*/
#include "opencv2/opencv.hpp"
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


//USE IF INPUT VECTOR TAKES UNSIGNED CHAR DATATYPE
void fill_buffer_with_mat_2(cv::Mat input, uint8_t* to_inp, int height, int width,int channels)
{
    //UNOPTIMIZED
    //To do: Find a memcpy version of this
    for(int i = 0; i < height; i++){
      int des_pos;
      for(int j = 0; j < width; j++){
        des_pos = (i * width + j) * channels;
        cv::Vec3b intensity = input.at<cv::Vec3b>(i, j);
        to_inp[des_pos] = (uint8_t) intensity.val[2]; //R
        to_inp[des_pos+1] = (uint8_t) intensity.val[1]; //G
        to_inp[des_pos+2] = (uint8_t) intensity.val[0]; //B
      }
    }
}

//USE IF INPUT VECTOR TAKES FLOAT DATATYPE
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

int main (int argc, char **argv)
{
  //Usage if in the Build Folder: liveSSD_pipeline ../<model name>.tflite

  if (argc != 2) {
    fprintf(stderr, "liveSSD_pipeline <tflite model relative path>\n");
    return 1;
  }

  const char* filename = argv[1];
  time_t timer_begin,timer_end;
	//raspicam::RaspiCam_Cv Camera;
	cv::Mat image;
  cv::Mat resized;
  int height = 300;
  int width = 300;
  int channels = 3;

  cv::VideoCapture Camera(0); //capture the video from webcam

  if ( !Camera.isOpened() )  // if not success, exit program
  {
       std::cout << "Cannot open the web cam" << std::endl;
       return -1;
  }

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

  //show empty output tensor (use for debugging)
  printf("=== Pre-invoke Interpreter Output State (first 6 rows) ===\n");
  int pre_out_idx = interpreter->outputs()[0];
  float* pre_out = interpreter->typed_tensor<float>(pre_out_idx);
  for(int n = 0; n < (6 * 33); n++){
    std::cout << pre_out[n] << " ";
    if((n%33 == 0) && (n != 0)){
      std::cout << "\n";
    }
  }

  //get pointer to input tensor
  int input = interpreter->inputs()[0];
  float* to_inp = interpreter->typed_tensor<float>(input);


  //camera currently set to capture frames of BGR color matrix format
  /*Camera.set( cv::CAP_PROP_FORMAT, CV_8UC3 );

  std::cout<< "\nOpening Camera..." << "\n";
	if (!Camera.open()) {
    std::cerr<<"Error opening the camera"<< "\n";
    return -1;
  }*/

  std::cout << "Starting display..."<< "\n";
  std::cout << "Press any key begin bounding box prediction." << "\n";

  cv::namedWindow("Camera View", cv::WINDOW_AUTOSIZE);

  //Test video-stream
  while(1){
    //Camera.grab();
		//Camera.retrieve(image);
    Camera >> image;
    cv::resize(image, resized, cv::Size(width,height));
		cv::imshow("Camera View",resized);
    if(cv::waitKey(30) >= 0) break;

  }
  std::cout << "Press ESC to end." << "\n";

  //Proper pipeline
  while(1){
    //Camera.grab();
    //Camera.retrieve(image);
    Camera >> image;
    cv::resize(image, resized, cv::Size(width,height));

    //Fill model input buffers with captured Mat frames
    fill_buffer_with_mat(resized,to_inp,height,width,channels);

    //Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    //Get pointer to output tensor
    int output_idx = interpreter->outputs()[0];
    float* output = interpreter->typed_tensor<float>(output_idx);

    //To do: Grab output predictions and feed to NMS
    //To do: Use NMS to cut down number of bounding boxes
    //To do: Draw bounding boxes on resized

    cv::imshow("Camera View",resized);

    char c = cv::waitKey(1);
    if(c == 27) break;
  }

  //Camera.release();
  return 0;

}
