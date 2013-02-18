#include "ClassifierBI.h"

ClassifierBI::ClassifierBI(IClassParams* oClassParamsP){
	oClassParams = oClassParamsP;
}

ClassifierBI::~ClassifierBI(){
}


void ClassifierBI::eval(int iPercCrossFold){

	std::cout << this->toString() << endl;

	int iDataSize = mCompleteData.rows;

	int iItemsInSet = iDataSize * iPercCrossFold / 100;
	for (int i = 0;i < mCompleteData.rows; i = i+ iItemsInSet)
	{
		createDataToEval(i, iItemsInSet);
		trainBI();
		testBI();
	}

	std::cout << "Well classifed: " <<oClassResults.TruePositive() << endl;
	std::cout << "Wrong classifed: " <<oClassResults.FalsePositive() << endl;

}

void ClassifierBI::createDataToEval(int iStartIndex, int iItemsInSet){
	mTestData = Mat(iItemsInSet, mCompleteData.cols, mCompleteData.type() );


	for(int i = iStartIndex; i < iStartIndex + iItemsInSet; i++){
		for (int j = 0; j<mCompleteData.cols; j ++)
		{
			mTestData.at<float>(i - iStartIndex, j) = mCompleteData.at<float>(i, j);
		}
	}


	vector<float>* vTrainingDataPoints = new vector<float>();

	for(int i = 0; i < iStartIndex; i++){
		for (int j = 0; j<mCompleteData.cols; j ++)
		{
			vTrainingDataPoints->push_back(mCompleteData.at<float>(i, j));
		}
	}

	for(int i = iStartIndex + iItemsInSet; i < mCompleteData.rows; i++){
		for (int j = 0; j<mCompleteData.cols; j ++)
		{
			float fValue = mCompleteData.at<float>(i, j);
			vTrainingDataPoints->push_back(mCompleteData.at<float>(i, j));
		}
	}



	mTrainingData = Mat(mCompleteData.rows - iItemsInSet, mCompleteData.cols, mCompleteData.type(), &((*vTrainingDataPoints)[0]));

}


void ClassifierBI::SplitTrainLabels(Mat mOriginalData, Mat &mTrainData, Mat &mLabelsData,  int iSplitIndex) 
{
	Range oRangeColsCompleteData =Range(0,iSplitIndex);
	mTrainData = mOriginalData(Range::all(), oRangeColsCompleteData);

	Range oRangeColsCompleteDataLabels =Range( iSplitIndex, iSplitIndex + 1);
	mLabelsData = mOriginalData(Range::all(), oRangeColsCompleteDataLabels);

}


cv::Mat ClassifierBI::CompleteData() const { return mCompleteData; }
void ClassifierBI::CompleteData(cv::Mat val) { mCompleteData = val; setParams();}

