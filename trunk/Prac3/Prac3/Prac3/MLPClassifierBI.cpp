#include "MLPClassifierBI.h"

MLPClassifierBI::MLPClassifierBI(MLPParamsBI* oMLPParamsBIP)
	:ClassifierBI(dynamic_cast<IClassParams*>(oMLPParamsBIP))
{
	Mat layers = Mat (4, 1, CV_32SC1 );
	layers.row (0) = Scalar (2);
	layers.row (1) = Scalar (10);
	layers.row (2) = Scalar (15);
	layers.row (3) = Scalar (1);
	oMLP.create (layers );


	oClassResults.TruePositive(0);
	oClassResults.FalsePositive(0);
}

MLPClassifierBI::~MLPClassifierBI(){

}

void MLPClassifierBI::setParams(){


	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;	
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams :: BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	oMLPParamsBI->MLPParamsField(params);

}



void MLPClassifierBI::trainBI(){

	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	CvANN_MLP_TrainParams oMLPParams = oMLPParamsBI->MLPParamsField();
	oMLP.train(mTrainingData, mTrainingDataLabels, Mat (), Mat (), oMLPParams );
}

void MLPClassifierBI::testBI(){


	/*for (int i = 0; i<mTestData.rows; i++)
	{

	Mat sampleRow = mTestData.row(i);
	Range oRangeColsTestData =Range(0, sampleRow.cols-1);
	Mat sampleMat = sampleRow(Range::all(), oRangeColsTestData);

	float response = oSVM.predict(sampleMat);

	float fLabel = sampleRow.at<float>(0, sampleRow.cols-1);

	if(fLabel== response){
	oClassResults.TruePositive(oClassResults.TruePositive()+1);
	}
	else{

	oClassResults.FalsePositive(oClassResults.FalsePositive()+1);
	}
	}*/

}

CvANN_MLP MLPClassifierBI::MLPObject() const { return oMLP; }

void MLPClassifierBI::MLPObject(CvANN_MLP val) { oMLP = val; }

string  MLPClassifierBI::toString(){
	string sResult =  "MLP Classifier";
	return sResult;
}



