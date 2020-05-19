#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
using namespace std;

vector<uint8_t> convert_mat(cv::Mat input, int height, int width,int channels)
{
  vector<uint8_t> output(height * width * channels);

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
  time_t timer_begin,timer_end;
	raspicam::RaspiCam_Cv Camera;
	cv::Mat image;
  cv::Mat resized;
  int height = 480;
  int width = 640;
  int channels = 3;


  //camera currently set to capture frames of BGR color matrix format
  Camera.set( cv::CAP_PROP_FORMAT, CV_8UC3 );

  cout<<"Opening Camera..."<<endl;
	if (!Camera.open()) {
    cerr<<"Error opening the camera"<<endl;
    return -1;
  }

  cout << "Starting display..."<< endl;
  cout << "Press ESC to begin bounding box prediction." << endl;

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
  cout << "Press ESC to end." << endl;

  //Proper pipeline
  while(1){
    Camera.grab();
    Camera.retrieve(image);
    cv::resize(image, resized, cv::Size(width,height));

    //Convert Mat resized to std::vector<uin8_t> to make compatible w/ tflite
    vector<uint8_t> input = convert_mat(resized,height,width,channels);

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
