#include "SVMClassifierBI.h"

SVMClassifierBI::SVMClassifierBI(SVMParamsBI* oSVMParamsBIP, Mat mCompleteData)
	:ClassifierBI(dynamic_cast<IClassParams*>(oSVMParamsBIP), mCompleteData)
{
	setParams();
	oSVM = SVM();
	oClassResults.TruePositive(0);
	oClassResults.FalsePositive(0);
}

SVMClassifierBI::~SVMClassifierBI(){

}

void SVMClassifierBI::setParams(){
	SVMParamsBI* oSVMParamsBIP = static_cast<SVMParamsBI*>(oClassParams);
	SVMParams oSVMParams = oSVMParamsBIP->SVMParamsField();

	Mat mCompleteDataInput, mCompleteDataLabels;

	SplitTrainLabels(mCompleteData, mCompleteDataInput, mCompleteDataLabels, mCompleteData.cols-1);

	oSVM.train_auto(mCompleteDataInput, mCompleteDataLabels, Mat(), Mat(), oSVMParams, PercCrossFold);
	oSVMParamsBIP->SVMParamsField(oSVM.get_params());

}



void SVMClassifierBI::trainBI(){



	SVMParamsBI* oSVMParamsBIP = static_cast<SVMParamsBI*>(oClassParams);
	SVMParams oSVMParams = oSVMParamsBIP->SVMParamsField();

	Mat mTrainingDataInput, mTrainingDataLabels;
	SplitTrainLabels(mTrainingData, mTrainingDataInput, mTrainingDataLabels, mTrainingData.cols-1);

	oSVM.train(mTrainingDataInput, mTrainingDataLabels, Mat(), Mat(), oSVMParams);
}

void SVMClassifierBI::testBI(){


	for (int i = 0; i<mTestData.rows; i++)
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
	}

}


SVM SVMClassifierBI::SVMObject() const { return oSVM; }

void SVMClassifierBI::SVMObject(SVM val) { oSVM = val; }



