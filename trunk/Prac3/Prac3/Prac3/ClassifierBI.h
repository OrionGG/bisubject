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
	ClassifierBI(IClassParams *oClassParamsP);
	~ClassifierBI();


	virtual void trainBI()=0;
	virtual void testBI()=0;
	virtual void setParams()= 0;
	virtual string toString()=0;

	void eval();

	virtual void prepareDataToEval( int i, int iItemsInSet )=0 ;
	

	void SplitTrainLabels(Mat mOriginalData, Mat &mTrainData, Mat &mLabelsData,  int iSplitIndex) ;


	map<int, Mat> CompleteData() const;
	void CompleteData(map<int, Mat> val, int iMinDataPerLabel);

	ClassResults getClassResults() const;
	void getClassResults(ClassResults val);


	void createDataToEval(int iStartIndex,int iItemsInSet, Mat vDataOneType, int iLabel, int iLabelsNumber);

protected:
	IClassParams* oClassParams;
	ClassResults oClassResults;
	int iPercCrossFold;
	Mat mTrainingData;
	Mat mTrainingDataLabels;
	Mat mTestData;
	Mat mTestDataLabels;
	map<int, Mat> hCompleteData;
	int iMinDataPerLabel;
	Mat mConfusiobMatrix;

};
#endif