#pragma once

class BaseLocation
{
public:
	BaseLocation();
	~BaseLocation();
	int verifySizes(RotatedRect rotated_rect);
	void tortuosity(Mat src, vector<RotatedRect> &rects, vector<Mat> &dst_plates);
	void safeRect(Mat src, RotatedRect &rect, Rect2f &dst_rect);
	void rotation(Mat src, Mat &dst, Size rect_size,
		Point2f center, double angle);
};

