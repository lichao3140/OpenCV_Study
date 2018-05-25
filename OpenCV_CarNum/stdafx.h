// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace ml;
using namespace std;


//根据训练样本决定的
//列向量 宽
#define ANN_COLS 8
//行向量  高
#define ANN_ROWS 16
#define HEIGHT 36
#define WIDTH 136

#include "CarPlateRecongize.h"

// TODO:  在此处引用程序需要的其他头文件
