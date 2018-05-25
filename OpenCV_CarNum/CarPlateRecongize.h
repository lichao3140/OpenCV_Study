#pragma once
#include "BaseLocation.h"
class CarPlateRecongize :
	public BaseLocation
{
public:
	CarPlateRecongize(const char * svm_path, const char * ann_path, const char * ann_zh_path);
	~CarPlateRecongize();
	string plateRecongize(Mat src, Mat &plate);
	void findPlateLocation(Mat src, Mat &dst);
	void plateLocate(Mat src, vector<Mat> &plates);
	void processMat(Mat src, Mat& dst, int blur_size, int close_w, int close_h);
	void getHOGFeatures(HOGDescriptor *hog, Mat image, Mat& features);
private:
	HOGDescriptor *svm_hog;
	HOGDescriptor *ann_hog;
	Ptr<SVM> svm;
	Ptr<ANN_MLP> ann;
	Ptr<ANN_MLP> ann_zh;
	void clearFixPoint(Mat &img);
	int verifyCharSizes(Mat &src);
	int getCityIndex(vector<Rect> &rects);
	void getChineseRect(Rect &src, Rect &des);
	string predict(vector<Mat> vec_chars, string &plate_result);
};

