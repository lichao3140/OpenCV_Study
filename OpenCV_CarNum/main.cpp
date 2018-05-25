// OpenCV_carPlate.cpp : 定义控制台应用程序的入口点。
//
#define _CRT_SECURE_NO_WARNINGS  //该语句必须在#include<stdio.h>之前，否则还会报错  
#include "stdafx.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


CarPlateRecongize * carPlateRecongize = 0;


void init() {
	carPlateRecongize = new CarPlateRecongize("HOG_SVM_DATA.xml",
		"HOG_ANN_DATA.xml",
		"HOG_ANN_ZH_DATA.xml");
}
int main()
{
	init();
	Mat img = imread("C:/Users/Administrator/Desktop/DL/benchi.jpg");
	Mat plate;
	cout << carPlateRecongize->plateRecongize(img, plate);

	waitKey(0);
	system("pause");
	return 0;
}

