#include "ClassifierBI.h"

ClassifierBI::ClassifierBI(IClassParams* oClassParamsP){
	oClassParams = oClassParamsP;
	iPercCrossFold = 10;
}

ClassifierBI::~ClassifierBI(){

}


ClassResults ClassifierBI::getClassResults() const { return oClassResults; }
void ClassifierBI::getClassResults(ClassResults val) { oClassResults = val; }

void ClassifierBI::CrossEval( int i, int iItemsInSet ) 
{
	mTrainingData = Mat(0,0,CV_32FC1);
	mTrainingDataLabels = Mat(0,0,CV_32FC1);
	mTestData =  Mat(0,0,CV_32FC1);
	mTestDataLabels =  Mat(0,0,CV_32FC1);
	for (map<int, Mat>::iterator it=hCompleteData.begin(); it!=hCompleteData.end(); ++it){
		Mat vDataOneType = it->second;

		createDataToEval(i,iItemsInSet, vDataOneType, it->first, hCompleteData.size());
	}

	trainBI();
	testBI();
	std::cout << "Well classifed: " <<oClassResults.TruePositive() << endl;
	std::cout << "Wrong classifed: " <<oClassResults.FalsePositive() << endl;
}

void ClassifierBI::eval(){


	int iItemsInSet = iMinDataPerLabel * iPercCrossFold / 100;
	for (int i = 0;i < 1/*iMinDataPerLabel*/; i = i+iItemsInSet)
	{
		 
		CrossEval(i, iItemsInSet);

	}

		std::cout << "Well classifed: " <<oClassResults.TruePositive() << endl;
		std::cout << "Wrong classifed: " <<oClassResults.FalsePositive() << endl;

	
	

}



void ClassifierBI::createDataToEval(int iStartIndex,int iItemsInSet, Mat vDataOneType, int iLabel, int iLabelsNumber){


	for(int i = iStartIndex; i < iStartIndex + iItemsInSet; i++){
		//mTestData.row(i - iStartIndex) = vDataOneType.row(i);
		//Mat mTestDataRow;
		//vDataOneType.row(i).convertTo(mTestDataRow, CV_32FC1);
		mTestData.push_back(vDataOneType.row(i));
		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
		mLabel.at<float>(0,iLabel) = 1.0;
		mTestDataLabels.push_back(mLabel);
	}



	for(int i = 0; i < iStartIndex; i++){		
		//Mat mTestDataRow;
		//vDataOneType.row(i).convertTo(mTestDataRow, CV_32FC1);
		mTrainingData.push_back(vDataOneType.row(i));
		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
		mLabel.at<float>(0,iLabel) = 1.0;
		mTrainingDataLabels.push_back(mLabel);
	}


	for(int i = iStartIndex + iItemsInSet; i < vDataOneType.rows; i++){
		//Mat mTestDataRow;
		//vDataOneType.row(i).convertTo(mTestDataRow, CV_32FC1);
		mTrainingData.push_back(vDataOneType.row(i));
		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
		mLabel.at<float>(0,iLabel) = 1.0;
		mTrainingDataLabels.push_back(mLabel);
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

