#include "ClassifierBI.h"

ClassifierBI::ClassifierBI(IClassParams* oClassParamsP){
	oClassParams = oClassParamsP;
	iPercCrossFold = 25;
}

ClassifierBI::~ClassifierBI(){

}


ClassResults ClassifierBI::getClassResults() const { return oClassResults; }
void ClassifierBI::getClassResults(ClassResults val) { oClassResults = val; }



void ClassifierBI::eval(){


	int iItemsInSet = iMinDataPerLabel * iPercCrossFold / 100;
	
	int iWellClassifed = 0;
	int iWrongClassifed = 0;
	for (int i = 0;i < iMinDataPerLabel-iItemsInSet/*1*/; i = i+iItemsInSet)
	{
		 
		prepareDataToEval(i, iItemsInSet);

		trainBI();
		testBI();

		std::cout << "Well classifed: " << oClassResults.TruePositive() << endl;
		std::cout << "Wrong classifed: " << oClassResults.FalsePositive() << endl;
		iWellClassifed += oClassResults.TruePositive();
		iWrongClassifed += oClassResults.FalsePositive();
		std::cout << "Efficiency: " << oClassResults.Efficiency() << endl;;

		oClassResults.TruePositive(0);
		oClassResults.FalsePositive(0);


	}


	std::cout << "Total Well classifed: " << iWellClassifed << endl;
	std::cout << "Total Wrong classifed: " << iWrongClassifed << endl;
	
	double dEfficiency = (double)iWellClassifed/(iWellClassifed + iWrongClassifed);
	std::cout << "Total Efficiency: " << dEfficiency << endl;;

	for(int i = 0; i < mConfusiobMatrix.rows; i++)
		for(int j = 0; j < mConfusiobMatrix.cols; j++){
			cout << "[" << i << "," << j << "]" << "= ";
			double dConfusionValue = mConfusiobMatrix.at<double>(i,j);
			cout << dConfusionValue << "; ";
		}

		cout << endl;
	
}



void ClassifierBI::createDataToEval(int iStartIndex,int iItemsInSet, Mat vDataOneType, int iLabel, int iLabelsNumber){

	//int iNumberTestData = vDataOneType.rows / iItemsInSet;
	//for(int i = 0; i < vDataOneType.rows; i = i+ iNumberTestData){
	//	mTestData.push_back(vDataOneType.row(i));
	//	Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
	//	mLabel.at<float>(0,iLabel) = 1.0;
	//	mTestDataLabels.push_back(mLabel);

	//	for(int j = i+1; j < i+iNumberTestData && j < vDataOneType.rows; j++){		
	//		mTrainingData.push_back(vDataOneType.row(j));
	//		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
	//		mLabel.at<float>(0,iLabel) = 1.0;
	//		mTrainingDataLabels.push_back(mLabel);
	//	}
	//}

	for(int i = iStartIndex; i < iStartIndex + iItemsInSet; i++){
		mTestData.push_back(vDataOneType.row(i));
		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
		mLabel.at<float>(0,iLabel) = 1.0;
		mTestDataLabels.push_back(mLabel);
	}



	for(int i = 0; i < iStartIndex; i++){		
		mTrainingData.push_back(vDataOneType.row(i));
		Mat mLabel(1,iLabelsNumber, CV_32FC1,Scalar::all(0.0));
		mLabel.at<float>(0,iLabel) = 1.0;
		mTrainingDataLabels.push_back(mLabel);
	}


	for(int i = iStartIndex + iItemsInSet; i < vDataOneType.rows; i++){
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
	iMinDataPerLabel = iMinDataPerLabelP;
}


//cv::Mat ClassifierBI::CompleteData() const { return hCompleteData; }
//void ClassifierBI::CompleteData(cv::Mat val) { hCompleteData = val; setParams();}

