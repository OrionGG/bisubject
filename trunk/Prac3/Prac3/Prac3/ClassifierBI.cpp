#include "ClassifierBI.h"

ClassifierBI::ClassifierBI(IClassParams* oClassParamsP){
	oClassParams = oClassParamsP;
	iPercCrossFold = 10;
}

ClassifierBI::~ClassifierBI(){

}



void ClassifierBI::eval(){


	int iItemsInSet = iMinDataPerLabel * iPercCrossFold / 100;
	for (int i = 0;i <(iMinDataPerLabel-iItemsInSet); i = i++)
	{

		mTrainingData = Mat();
		mTrainingDataLabels = Mat();
		mTestData = Mat();
		mTestDataLabels = Mat();
		for (map<int, Mat>::iterator it=hCompleteData.begin(); it!=hCompleteData.end(); ++it){
			Mat vDataOneType = it->second;
			
			createDataToEval(i,iItemsInSet, vDataOneType, it->first);
		}


			trainBI();
			testBI();
	}

		std::cout << "Well classifed: " <<oClassResults.TruePositive() << endl;
		std::cout << "Wrong classifed: " <<oClassResults.FalsePositive() << endl;

	
	

}

//
//void ClassifierBI::createDataToEval(int iStartIndex, int iItemsInSet){
//	mTestData = Mat(iItemsInSet, mCompleteData.cols, mCompleteData.type() );
//
//
//	for(int i = iStartIndex; i < iStartIndex + iItemsInSet; i++){
//		for (int j = 0; j<mCompleteData.cols; j ++)
//		{
//			mTestData.at<float>(i - iStartIndex, j) = mCompleteData.at<float>(i, j);
//		}
//	}
//
//
//	vector<float>* vTrainingDataPoints = new vector<float>();
//
//	for(int i = 0; i < iStartIndex; i++){
//		for (int j = 0; j<mCompleteData.cols; j ++)
//		{
//			vTrainingDataPoints->push_back(mCompleteData.at<float>(i, j));
//		}
//	}
//
//	for(int i = iStartIndex + iItemsInSet; i < mCompleteData.rows; i++){
//		for (int j = 0; j<mCompleteData.cols; j ++)
//		{
//			float fValue = mCompleteData.at<float>(i, j);
//			vTrainingDataPoints->push_back(mCompleteData.at<float>(i, j));
//		}
//	}
//
//
//
//	mTrainingData = Mat(mCompleteData.rows - iItemsInSet, mCompleteData.cols, mCompleteData.type(), &((*vTrainingDataPoints)[0]));
//
//}






void ClassifierBI::createDataToEval(int iStartIndex,int iItemsInSet, Mat vDataOneType, int iLabel){


	for(int i = iStartIndex; i < iStartIndex + iItemsInSet; i++){
		//mTestData.row(i - iStartIndex) = vDataOneType.row(i);
		mTestData.push_back(vDataOneType.row(i));
		mTestDataLabels.push_back(iLabel);
	}



	for(int i = 0; i < iStartIndex; i++){
		mTrainingData.push_back(vDataOneType.row(i));
		mTrainingDataLabels.push_back(iLabel);
	}


	for(int i = iStartIndex + iItemsInSet; i < vDataOneType.rows; i++){
		mTrainingData.push_back(vDataOneType.row(i));
		mTrainingDataLabels.push_back(iLabel);
	}	
}


void ClassifierBI::SplitTrainLabels(Mat mOriginalData, Mat &mTrainData, Mat &mLabelsData,  int iSplitIndex) 
{
	Range oRangeColsCompleteData =Range(0,iSplitIndex);
	mTrainData = mOriginalData(Range::all(), oRangeColsCompleteData);

	Range oRangeColsCompleteDataLabels =Range( iSplitIndex, iSplitIndex + 1);
	mLabelsData = mOriginalData(Range::all(), oRangeColsCompleteDataLabels);

}

map<int, Mat> ClassifierBI::CompleteData() const { return hCompleteData; }

void ClassifierBI::CompleteData(map<int, Mat> val, int iMinDataPerLabelP) { 
	
	hCompleteData = val; 
	setParams(); 
	iMinDataPerLabel = iMinDataPerLabelP;
}


//cv::Mat ClassifierBI::CompleteData() const { return hCompleteData; }
//void ClassifierBI::CompleteData(cv::Mat val) { hCompleteData = val; setParams();}

