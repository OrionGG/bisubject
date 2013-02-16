#pragma once
#ifndef SVMParamsBI_H
#define SVMParamsBI_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"

#include "IClassParams.h"

using namespace cv;



class SVMParamsBI:public IClassParams {
public:
	SVMParamsBI();
	~SVMParamsBI();


	cv::SVMParams SVMParamsField() const { return oSVMParmas; }
	void SVMParamsField(cv::SVMParams val) { oSVMParmas = val; }

private:
	// Set up SVM's parameters
	SVMParams oSVMParmas;
};
#endif