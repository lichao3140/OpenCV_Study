#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

//保存拼接的原始图像向量
vector<Mat> imgs;

//导入所有原始拼接图像函数
void parseCmdArgs(int argc, char** argv);

//  stitcher全景图拼接
int main(int argc, char** argv) {
	//导入拼接图像
	parseCmdArgs(argc, argv);
	//最后拼接图片
	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(imgs, pano);//拼接
	if (status != Stitcher::OK) //判断拼接是否成功
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}
	namedWindow("全景拼接", 0);
	imshow("全景拼接", pano);
	imwrite("D:\\全景拼接.jpg", pano);
	waitKey(0);
	return 0;
}

//导入所有原始拼接图像函数
void parseCmdArgs(int argc, char** argv) 
{
	for (int i = 1; i < argc; i++)
	{
		Mat img = imread(argv[i]);
		if (img.empty())
		{
			cout << "Can't read image '" << argv[i] << "'\n";
		}
		imgs.push_back(img);
	}

}