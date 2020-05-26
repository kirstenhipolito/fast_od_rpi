#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

#include "decode_detections.hpp"

using namespace std;
using namespace Eigen;

void draw_bounding_boxes(cv::Mat input, const Ref<const MatrixXf>& boxes)
{
  int num_boxes = boxes.rows();

  for (int i = 0; i < num_boxes; i++)
  {
    cv::Point topleft = cv::Point(boxes(i,2),boxes(i,3));
    cv::Point bottomright = cv::Point(boxes(i,4),boxes(i,5));
    cv::rectangle(input, topleft, bottomright, cv::Scalar(0,0,255));
  }
}

int main()
{

  cv::Mat image;
  cv::Mat resized;
  int height = 300;
  int width = 300;
  string filename = "silicon_valley.jpg";

  image = cv::imread(filename,1);
  cv::resize(image, resized, cv::Size(width,height));

  if(!image.data) {
      cout << "Can't open file " << filename << '\n';
      return -1;
  }

  const int y_pred_rows = 2006;
	const int y_pred_cols = 33;
  MatrixXf y_pred(y_pred_rows, y_pred_cols);
  ifstream myReadFile;

  myReadFile.open("silicon_y_pred_raw.txt");

  while (!myReadFile.eof()){
    for(int i = 0; i < y_pred_rows; i++){
      for (int j = 0; j < y_pred_cols; j++){
        myReadFile >> y_pred(i,j);
      }
    }
  }

  MatrixXf vec_boxes = decode_detections(y_pred, 0.3, 0.45, 4, 300, 300);
  cout << "vec_boxes:\n" << vec_boxes << endl;

  cv::Mat original_img = resized.clone();
  draw_bounding_boxes(resized,vec_boxes);

  cv::namedWindow("original resized", cv::WINDOW_AUTOSIZE);
  cv::imshow("original resized", original_img);
  cv::namedWindow("with bounding boxes", cv::WINDOW_AUTOSIZE);
  cv::imshow("with bounding boxes", resized);

  // Wait until the user presses any key
  cv::waitKey(0);


  return 0;
}
