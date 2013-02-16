#pragma once
#ifndef CLASSIFIERBI_H
#define CLASSIFIERBI_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"


#include "IClassParams.h"
#include "ClassResults.h"


using namespace std;
using namespace cv;

class ClassifierBI{
public:
	ClassifierBI(IClassParams *oClassParamsP, Mat mCompleteData);
	~ClassifierBI();


	virtual void trainBI()=0;
	virtual void testBI()=0;
	virtual void setParams()= 0;

	void eval(int iPercCrossFold);

	void SplitTrainLabels(Mat mOriginalData, Mat &mTrainData, Mat &mLabelsData,  int iSplitIndex) ;

private:
	void createDataToEval(int iStartIndex, int iItemsInSet);

protected:
	IClassParams* oClassParams;
	ClassResults oClassResults;
	Mat mTrainingData;
	Mat mTestData;
	Mat mCompleteData;

};
#endif