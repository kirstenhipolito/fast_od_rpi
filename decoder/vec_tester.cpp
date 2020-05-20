#include <iostream>
#include <fstream>
#include <Eigen/Dense>
// #include <chrono>
#include <vector>
// #include <opencv2/opencv.hpp>

// #include "utils.hpp"
// #include "vectorized_nms.hpp"
#include "decode_detections.hpp"
// #include "nms.hpp"



// using namespace cv;
using namespace std;
// using namespace std::chrono;
using namespace Eigen;


int main()
{
  // initialization
  // Mat imgBefore(Size(500, 500), DataType<float>::type);
  // Mat imgBefore("silicon_valley.jpg");

  // Mat imgAfter = imgBefore.clone();
  // float threshold	= 0.5;


	const int y_pred_rows = 2006;
	const int y_pred_cols = 33;



  MatrixXf y_pred(y_pred_rows, y_pred_cols);
  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_raw.txt");

  while (!myReadFile.eof()){
    for(int i = 0; i < y_pred_rows; i++){
      for (int j = 0; j < y_pred_cols; j++){
        // myReadFile >> y_pred[i][j];
        myReadFile >> y_pred(i,j);
      }
    }
  }


	MatrixXf vec_boxes = decode_detections(y_pred, 0.3, 0.45, 4, 300, 300);
  cout << "vec_boxes:\n" << vec_boxes << endl;

  // before
  // DrawRectangles(imgBefore, vec_boxes);
  // imshow("Before", imgBefore);
  



  //
  //  TESTING VECTORIZED NMS vs HARDCODED NMS
  //
	// const int nms_rows = 52;
	// const int nms_cols = 5;

  // MatrixXf vec_boxes(nms_rows, nms_cols);
  // ifstream myReadFile;
  // myReadFile.open("tflite_y_pred_before_nms.txt");


  // while (!myReadFile.eof()){
  //   for(int i = 0; i < nms_rows; i++){
  //     for (int j = 0; j < nms_cols; j++){
  //       myReadFile >> vec_boxes(i,j);
  //       // myReadFile >> boxes[i][j];
  //     }
  //   }
  // }




	// vector<Rect> vectorized_reducedRectangle;

  // auto vec_start = high_resolution_clock::now();

	// for(int i=0; i<10000; i++){
  //   vectorized_reducedRectangle = vectorized_nms(vec_boxes, threshold);
	// }

  // auto vec_stop = high_resolution_clock::now();
  // auto vec_duration = duration_cast<milliseconds>(vec_stop - vec_start);
	// // float vec_time = vec_duration.count()/1000000;
  // cout << "C++ vectorized_nms duration: " << vec_duration.count() << "ms" << endl;   
  // DrawRectangles(imgAfter, vectorized_reducedRectangle);
  // imshow("Vectorized NMS", imgAfter);

  // waitKey(0);
}