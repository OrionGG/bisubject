#pragma once
#ifndef NBCLASSIFIERBI_H
#define NBCLASSIFIERBI_H
#include "opencv2/ml/ml.hpp"

#include "ClassifierBI.h"
#include "IClassParams.h"
#include "NBParamsBI.h"

using namespace cv;

class NBClassifierBI: public ClassifierBI{
public:
	NBClassifierBI(NBParamsBI* oNBParamsBIP, Mat mCompleteData);
	~NBClassifierBI();


	void trainBI();
	void testBI();

private:
	NormalBayesClassifier oNormalBayesClassifier;

	void setParams();
};
#endif