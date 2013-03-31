#pragma once
#ifndef MLPCLASSIFIERBI_H
#define MLPCLASSIFIERBI_H
#include <ctime>
#include "opencv2/ml/ml.hpp"

#include "ClassifierBI.h"
#include "IClassParams.h"
#include "MLPParamsBI.h"
#include "Global.h"

using namespace cv;

class MLPClassifierBI:public ClassifierBI{
public:
	MLPClassifierBI(MLPParamsBI* oMLPParamsBIP, int iInputNumberP, int iOutputNumberP);
	~MLPClassifierBI();

	void prepareDataToEval( int i, int iItemsInSet ) ;

	void trainBI();
	void testBI();
	string toString();



	CvANN_MLP MLPObject() const;
	void MLPObject(CvANN_MLP  val);

protected:
	void setParams();
	CvANN_MLP oMLP;
	int iInputNumber;
	int iOutputNumber;

};
#endif