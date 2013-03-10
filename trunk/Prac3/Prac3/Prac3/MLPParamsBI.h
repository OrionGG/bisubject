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



class MLPParamsBI:public IClassParams {
public:
	MLPParamsBI();
	~MLPParamsBI();

	
	CvANN_MLP_TrainParams MLPParamsField() const { return oMLPParams; }
	void MLPParamsField(CvANN_MLP_TrainParams val) { oMLPParams = val; }

private:
	// Set up SVM's parameters
	CvANN_MLP_TrainParams oMLPParams;
};
#endif