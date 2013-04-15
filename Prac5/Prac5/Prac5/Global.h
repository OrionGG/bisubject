//#pragma once
#ifndef GLOBAL_H_
#define GLOBAL_H_
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;


extern const char* ERROROPENFILE;

class Global{
	public:
		static Mat openImage(const char*  sImageFile);
		static Mat openImage(const char*  sImageFile, int );
		

	private:
};
#endif