#pragma once
#ifndef PCAMLPCLASSIFIERBI_H
#define PCAMLPCLASSIFIERBI_H


#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include "MLPClassifierBI.h"
#include "ClassifierBI.h"


using namespace cv;
using namespace std;
using namespace boost::filesystem;

class PCAMLPClassifierBI:public MLPClassifierBI{
public:
	PCAMLPClassifierBI(MLPParamsBI* oMLPParamsBIP, int iInputNumberP, int iOutputNumberP, int iNumComponentsP);
	~PCAMLPClassifierBI();

	void prepareDataToEval( int i, int iItemsInSet ) ;

	void CompleteData(map<int, Mat> val, int iMinDataPerLabelP);

	void setParams();
private:
	Mat norm_0_255(const Mat& src);
	int iNumComponents;
	Mat mCompleteData;
	PCA oPCA;

};
#endif