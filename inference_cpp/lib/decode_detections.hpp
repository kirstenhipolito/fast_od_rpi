#ifndef DECODE_DECTECTIONS_HPP__
#define DECODE_DECTECTIONS_HPP__ 

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>


using namespace Eigen;
using namespace std;
// using namespace std::chrono;
using std::vector;
// using cv::Rect;
// using cv::Point;


MatrixXf decode_detections(const MatrixXf & y_pred, const float & confidence_thresh=0.3, const float & iou_threshold=0.45, const int & top_k=200, const int & img_height=300, const int & img_width=300);

void draw_bounding_boxes(cv::Mat input, const Ref<const MatrixXf>& boxes);

void draw_bounding_boxes_save(cv::Mat input, const Ref<const MatrixXf>& boxes, string img_path)

MatrixXf convert_coordinates(const MatrixXf & matrix);

MatrixXf vectorized_nms(const MatrixXf & boxes, const float & iou_thresh);

VectorXi argsort_eigen(VectorXf & vec);

void append_int_eigen(VectorXi & vect, int & value);

VectorXf extract_values(VectorXf & vec, VectorXi & idxs);

VectorXf max_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

VectorXf min_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

#endif // DECODE_DETECTIONS_HPP__ 