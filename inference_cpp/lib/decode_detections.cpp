#include "decode_detections.hpp"

MatrixXf decode_detections(const MatrixXf & y_pred, const float & confidence_thresh, const float & iou_threshold, const int & top_k, const int & img_height, const int & img_width){
	typedef Matrix <bool, Dynamic, Dynamic> MatrixXb;
	const int y_pred_rows = y_pred.rows();
	const int y_pred_cols = y_pred.cols();
	const int y_raw_cols = y_pred_cols - 8;
	const int n_classes = y_raw_cols-4;

	MatrixXf y_pred_decoded_raw(y_pred_rows,y_raw_cols);



	y_pred_decoded_raw = y_pred.block(0,0,y_pred_rows,y_raw_cols);
	y_pred_decoded_raw.block(0,y_raw_cols-2,y_pred_rows,2) = (y_pred_decoded_raw.block(0,y_raw_cols-2,y_pred_rows,2).array() * y_pred.block(0,y_pred_cols-2,y_pred_rows,2).array()).exp();
	y_pred_decoded_raw.block(0,y_raw_cols-2,y_pred_rows,2) = y_pred_decoded_raw.block(0,y_raw_cols-2,y_pred_rows,2).array() * y_pred.block(0,y_pred_cols-6,y_pred_rows,2).array();
	y_pred_decoded_raw.block(0,y_raw_cols-4,y_pred_rows,2) = y_pred_decoded_raw.block(0,y_raw_cols-4,y_pred_rows,2).array() * y_pred.block(0,y_pred_cols-4,y_pred_rows,2).array() * y_pred.block(0,y_pred_cols-6,y_pred_rows,2).array();
	y_pred_decoded_raw.block(0,y_raw_cols-4,y_pred_rows,2) = y_pred_decoded_raw.block(0,y_raw_cols-4,y_pred_rows,2).array() + y_pred.block(0,y_pred_cols-8,y_pred_rows,2).array();

	y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw);

	// Normalizing coordinates
	y_pred_decoded_raw.col(y_raw_cols-1) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-3) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-2) *= img_width;
	y_pred_decoded_raw.col(y_raw_cols-4) *= img_width;


	Matrix <float, Dynamic, 6> pred;
	for (int class_id = 1; class_id < n_classes; class_id++){
		MatrixXf single_class(y_pred_rows,5);
		single_class.col(0) = y_pred_decoded_raw.col(class_id);
		single_class.block(0,1,y_pred_rows,4) = y_pred_decoded_raw.block(0,y_raw_cols-4,y_pred_rows,4);

		VectorXf Thresh_Met(y_pred_rows,1);
		Thresh_Met = single_class.col(0);
		Thresh_Met = (Thresh_Met.array() > confidence_thresh).select(Thresh_Met, 0);
		MatrixXb non_zeros(y_pred_rows,1);
		non_zeros = Thresh_Met.cast<bool>().rowwise().any();
		MatrixXf threshold_met(non_zeros.count(),5);

		int j=0;
		for (int i=0 ; i<y_pred_rows ; ++i){
			if (non_zeros(i)){
				threshold_met.row(j++) = single_class.row(i);
			}
		}


		MatrixXf maxima = vectorized_nms(threshold_met, iou_threshold);
		const int maxima_rows = maxima.rows();

		// Including class in output
		MatrixXf maxima_output(maxima_rows,6);
		maxima_output.col(0) = ArrayXf::Zero(maxima_rows) + class_id;
		maxima_output.block(0,1,maxima_rows,5) = maxima;

		if(maxima_rows!=0){
			int pred_rows = pred.rows();
			pred.conservativeResize(pred_rows+maxima_rows,NoChange);
			pred.block(pred_rows,0,maxima_rows,6) = maxima_output;
		}
	}
	Matrix <float,Dynamic,6> y_pred_decoded;

	if (pred.rows()>top_k){
		// Get indices of top_k predictions
		VectorXi idxs = VectorXi::LinSpaced(pred.rows(), 0, pred.rows()-1);
		sort(idxs.data(), idxs.data()+idxs.size(),[&pred](int i1, int i2) {return pred.data()[i1] < pred.data()[i2];});
		idxs.conservativeResize(top_k);
		cout << "final idxs:\n" << idxs << endl;

		// Keep predictions with those indices
		y_pred_decoded.conservativeResize(top_k,NoChange);
		for (int i=0; i<top_k; i++){
			y_pred_decoded.row(i) = pred.row(idxs.data()[i]);
		}
	}
	else{
		y_pred_decoded.conservativeResize(pred.rows(),NoChange);
		y_pred_decoded = pred;
	}



	// vector<vector<float>> boxes(pred.rows(),vector<float>(5));
	// float array[pred.rows()][5];

	// // converted to array
	// for (int i=0 ; i<pred.rows() ; i++){
	// 	Map<RowVectorXf>(&array[i][0], 1, 5) = pred.row(i);
	// }
	// // converted to 2dvec
    // for(int i = 0; i < pred.rows(); i++){
    //   for (int j = 0; j < 5; j++){
    //     boxes[i][j] = array[i][j];
    //   }
    // }



	return y_pred_decoded;
}

void draw_bounding_boxes(cv::Mat input, const Ref<const MatrixXf>& boxes, int mode){
  //classificaions of the PASCAL VOC 2012 Dataset
  string classNames[21] = {"background","aeroplane", "bicycle", "bird",
          "boat","bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};

	if (mode == 2)
	{
		classNames[1] = "person";
	}

  //get number of bounding boxes
  int num_boxes = boxes.rows();

  //draw all boxes and classifications on image (all red boxes/text for now)
  for (int i = 0; i < num_boxes; i++)
  {
    //pixels above top left corner of box on the y-axis
    int confPointOffset = 3;
    int classPointOffset = 16;

    //scale of text size compared to default
    double textScale = 0.35;

    int classIndex = (int) boxes(i,0);
    string confidenceValue = to_string(boxes(i,1));

    cv::Point topleft = cv::Point(boxes(i,2),boxes(i,3));
    cv::Point bottomright = cv::Point(boxes(i,4),boxes(i,5));
    cv::Point confPoint = cv::Point(boxes(i,2),boxes(i,3) - confPointOffset);
    cv::Point classPoint = cv::Point(boxes(i,2),boxes(i,3) - classPointOffset);

    cv::rectangle(input, topleft, bottomright, cv::Scalar(0,0,255));
    cv::putText(input,classNames[classIndex],classPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
    cv::putText(input,confidenceValue,confPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
  }
}

void draw_bounding_boxes_custom_dim(cv::Mat input, const Ref<const MatrixXf>& boxes, int mode, int frame_width, int frame_height){
  //classificaions of the PASCAL VOC 2012 Dataset
  string classNames[21] = {"background","aeroplane", "bicycle", "bird",
          "boat","bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};

	if (mode == 2)
	{
		classNames[1] = "person";
	}

  //get number of bounding boxes
  int num_boxes = boxes.rows();

  //draw all boxes and classifications on image (all red boxes/text for now)
  for (int i = 0; i < num_boxes; i++)
  {
    //pixels above top left corner of box on the y-axis
    int confPointOffset = 3;
    int classPointOffset = 16;

    //scale of text size compared to default
    double textScale = 0.5;

    int classIndex = (int) boxes(i,0);
    string confidenceValue = to_string(boxes(i,1));

    float frame_width_multiplier = frame_width/300;
    float frame_height_multiplier = frame_height/300;

    // cv::Point topleft = cv::Point(boxes(i,2),boxes(i,3));
    // cv::Point bottomright = cv::Point(boxes(i,4),boxes(i,5));
    cv::Point topleft = cv::Point(boxes(i,2)*frame_width_multiplier,boxes(i,3)*frame_height_multiplier);
    cv::Point bottomright = cv::Point(boxes(i,4)*frame_width_multiplier,boxes(i,5)*frame_height_multiplier);

    cv::Point confPoint = cv::Point(boxes(i,2),boxes(i,3) - confPointOffset);
    cv::Point classPoint = cv::Point(boxes(i,2),boxes(i,3) - classPointOffset);

    cv::rectangle(input, topleft, bottomright, cv::Scalar(0,0,255));
    cv::putText(input,classNames[classIndex],classPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
    cv::putText(input,confidenceValue,confPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
  }
}

void draw_bounding_boxes_save(cv::Mat input, const Ref<const MatrixXf>& boxes, string img_path, int mode){
  //classificaions of the PASCAL VOC 2012 Dataset
  string classNames[21] = {"background","aeroplane", "bicycle", "bird",
          "boat","bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};

	if (mode == 2)
	{
		classNames[1] = "person";
	}

  //get number of bounding boxes
  int num_boxes = boxes.rows();

  //draw all boxes and classifications on image (all red boxes/text for now)
  for (int i = 0; i < num_boxes; i++)
  {
    //pixels above top left corner of box on the y-axis
    int confPointOffset = 3;
    int classPointOffset = 16;

    //scale of text size compared to default
    double textScale = 0.35;

    int classIndex = (int) boxes(i,0);
    string confidenceValue = to_string(boxes(i,1));

    cv::Point topleft = cv::Point(boxes(i,2),boxes(i,3));
    cv::Point bottomright = cv::Point(boxes(i,4),boxes(i,5));
    cv::Point confPoint = cv::Point(boxes(i,2),boxes(i,3) - confPointOffset);
    cv::Point classPoint = cv::Point(boxes(i,2),boxes(i,3) - classPointOffset);

    cv::rectangle(input, topleft, bottomright, cv::Scalar(0,0,255));
    cv::putText(input,classNames[classIndex],classPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
    cv::putText(input,confidenceValue,confPoint,cv::FONT_HERSHEY_SIMPLEX,textScale,cv::Scalar(0,0,255));
  }

  cv::imwrite(img_path, input);
}

MatrixXf convert_coordinates(const MatrixXf & matrix){
	MatrixXf converted = matrix;
	int start_index = matrix.cols();
	converted.col(start_index-4) = matrix.col(start_index-4) - matrix.col(start_index-2) / 2;
	converted.col(start_index-3) = matrix.col(start_index-3) - matrix.col(start_index-1) / 2;
	converted.col(start_index-2) = matrix.col(start_index-4) + matrix.col(start_index-2) / 2;
	converted.col(start_index-1) = matrix.col(start_index-3) + matrix.col(start_index-1) / 2;
	return converted;
}

MatrixXf vectorized_nms(const MatrixXf & pred, const float & iou_threshold){
	VectorXf conf = pred.col(0);
    VectorXf x1 = pred.col(1);
    VectorXf y1 = pred.col(2);
    VectorXf x2 = pred.col(3);
    VectorXf y2 = pred.col(4);
    VectorXf area;



	// Compute Area
    area = ((x2 - x1).array() + 1) * ((y2 - y1).array() + 1);




	// FIX THIS: ARGSORT CONF TO GET IDXS

	VectorXi idxs = argsort_eigen(conf);


	// BEGIN NMS LOOP
	int last;
	int i;
	VectorXi pick;


	while (idxs.size() > 0){

		last = idxs.size() - 1;
		i = idxs[last];

		// Remove last from idxs
		VectorXi idxs_no_last;
		idxs_no_last = idxs;
		idxs_no_last.conservativeResize(idxs_no_last.rows()-1,NoChange);


		append_int_eigen(pick, i);
		// cout << "pick: " << pick << endl;

		VectorXf extracted_x1 = extract_values(x1, idxs_no_last);
		VectorXf xx1 = max_eigen(x1, i, extracted_x1);

		VectorXf extracted_y1 = extract_values(y1, idxs_no_last);
		VectorXf yy1 = max_eigen(y1, i, extracted_y1);

		VectorXf extracted_x2 = extract_values(x2, idxs_no_last);
		VectorXf xx2 = min_eigen(x2, i, extracted_x2);

		VectorXf extracted_y2 = extract_values(y2, idxs_no_last);
		VectorXf yy2 = min_eigen(y2, i, extracted_y2);



		// Compute Height and Width
		VectorXf w = (xx2 - xx1).array() + 1;
		w = (w.array() >= 0).select(w, 0);

		VectorXf h = (yy2 - yy1).array() + 1;
		h = (h.array() >= 0).select(h, 0);


		VectorXf extracted_area = extract_values(area, idxs_no_last);
		VectorXf overlap = (w.array()*h.array()) / extracted_area.array();

		VectorXf bool_keep = overlap;
		bool_keep = (overlap.array() <= iou_threshold).select(bool_keep, 1);


		int num_keep = (bool_keep.array()<1).count();


		VectorXi tmp_idxs = idxs;
		idxs.conservativeResize(num_keep, NoChange);


		int m=0;
		for (int n=0; n<bool_keep.size(); ++n){
			if(bool_keep(n)<1){
				idxs.row(m++) = tmp_idxs.row(n);
			}
		}
		// cout << "idxs:\n" << idxs << endl;
		// cout << "idxs.size(): " << idxs.size() << endl;


	}

	// cout << "pick:\n" << pick << endl;
	// cout << "pred shape: (" << pred.rows() << "," << pred.cols() << ")\n";

	int n_finalboxes = pick.rows();

	MatrixXf filtered(n_finalboxes,5);
	for (int x=0; x<n_finalboxes; x++){
		filtered.row(x) = pred.row(pick.data()[x]);
	}

	//
	//	Conversion to array
	//
	vector<vector<float>> final_boxes(n_finalboxes,vector<float>(5));
	float array_final[n_finalboxes][5];

	// converted to array
	for (int i=0 ; i<n_finalboxes ; i++){
		Map<RowVectorXf>(&array_final[i][0], 1, 5) = filtered.row(i);
	}
	// converted to 2dvec
	for(int i = 0; i < n_finalboxes; i++){
		for (int j = 0; j < 5; j++){
		final_boxes[i][j] = array_final[i][j];
		}
	}


    // auto boxes_rect = VecBoxesToRectangles(final_boxes);


    return filtered;
}

VectorXi argsort_eigen(VectorXf & vec){
	// Initialize indices
	VectorXi idxs = VectorXi::LinSpaced(vec.size(), 0, vec.size()-1);
	// Sort indices
	sort(idxs.data(), idxs.data()+idxs.size(),[&vec](int i1, int i2) {return vec.data()[i1] < vec.data()[i2];});

	return idxs;
}


void append_int_eigen(VectorXi & vect, int & value)
{
    int row = vect.rows();
	vect.conservativeResize(row + 1, NoChange);
    vect.row(row) << value;
}

VectorXf extract_values(VectorXf & vec, VectorXi & idxs){
	int n_idxs = idxs.size();
	VectorXf resultVec(n_idxs);
	for (int i=0; i<n_idxs; i++){
		resultVec.row(i) << vec.row(idxs.data()[i]);
	}
  return resultVec;
}

VectorXf max_eigen(VectorXf & vec1, int & i, VectorXf & vec2){
	VectorXf maxVec = vec2;
	float x = vec1.data()[i];
	maxVec = (maxVec.array() >= x).select(maxVec, x);
	return maxVec;
}

VectorXf min_eigen(VectorXf & vec1, int & i, VectorXf & vec2){
	VectorXf maxVec = vec2;
	float x = vec1.data()[i];
	maxVec = (maxVec.array() <= x).select(maxVec, x);
	return maxVec;
}
// vector<Rect> VecBoxesToRectangles(const vector<vector<float>> & boxes)
// {
//   vector<Rect> rectangles;
//   vector<float> box;

//   for (const auto & box: boxes)
//     rectangles.push_back(Rect(Point(box[1], box[2]), Point(box[3], box[4])));

//   return rectangles;
// }
