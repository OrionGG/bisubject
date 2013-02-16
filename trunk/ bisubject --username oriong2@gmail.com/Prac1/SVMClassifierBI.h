#pragma once
#ifndef SVMCLASSIFIERBI_H
#define SVMCLASSIFIERBI_H
#include "opencv2/ml/ml.hpp"

#include "ClassifierBI.h"
#include "IClassParams.h"
#include "SVMParamsBI.h"
#include "Global.h"

using namespace cv;

class SVMClassifierBI:public ClassifierBI{
public:
	SVMClassifierBI(SVMParamsBI* oSVMParamsBIP, Mat mCompleteData);
	~SVMClassifierBI();



	void trainBI();
	void testBI();



	SVM SVMObject() const;
	void SVMObject(SVM  val);

private:
	SVM oSVM;
	void setParams();

};
#endif