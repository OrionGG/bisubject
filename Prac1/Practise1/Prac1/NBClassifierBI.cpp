#include "NBClassifierBI.h"

NBClassifierBI::NBClassifierBI(NBParamsBI* oNBParamsBIP):
ClassifierBI(dynamic_cast<IClassParams*>(oNBParamsBIP)){

	setParams();
	oClassResults.TruePositive(0);
	oClassResults.FalsePositive(0);
}

NBClassifierBI::~NBClassifierBI(){
}


void NBClassifierBI::trainBI(){


	Mat mTrainingDataInput, mTrainingDataLabels;
	SplitTrainLabels(mTrainingData, mTrainingDataInput, mTrainingDataLabels, mTrainingData.cols-1);
	oNormalBayesClassifier.train(mTrainingDataInput, mTrainingDataLabels, Mat(), Mat(), false);

}

void NBClassifierBI::testBI(){
	for (int i = 0; i<mTestData.rows; i++)
	{	
		Mat sampleRow = mTestData.row(i);
		Range oRangeColsTestData =Range(0, sampleRow.cols-1);
		Mat sampleMat = sampleRow(Range::all(), oRangeColsTestData);

		float response = oNormalBayesClassifier.predict(sampleMat);

		float fLabel = sampleRow.at<float>(0, sampleRow.cols-1);

		if(fLabel== response){
			oClassResults.TruePositive(oClassResults.TruePositive()+1);
		}
		else{

			oClassResults.FalsePositive(oClassResults.FalsePositive()+1);
		}
	}
}

void NBClassifierBI::setParams(){

}

string  NBClassifierBI::toString(){
	string sResult =  "NB Classifier";
	return sResult;
}