#include "MLPClassifierBI.h"

MLPClassifierBI::MLPClassifierBI(MLPParamsBI* oMLPParamsBIP, int iInputNumberP, int iOutputNumberP)
	:ClassifierBI(dynamic_cast<IClassParams*>(oMLPParamsBIP))
{
	iInputNumber = iInputNumberP;
	iOutputNumber =iOutputNumberP;

	oClassResults.TruePositive(0);
	oClassResults.FalsePositive(0);


	mConfusiobMatrix = Mat(iOutputNumber, iOutputNumber, CV_64FC1, Scalar::all(0.0));
}

MLPClassifierBI::~MLPClassifierBI(){

}

void MLPClassifierBI::setParams(){

	if(oMLP.get_layer_count() == 0){
		oMLP = CvANN_MLP();
	}
	//Mat layers = Mat (2, 1, CV_32SC1 );
	Mat layers = Mat (3, 1, CV_32SC1 );
	layers.row (0) = Scalar (iInputNumber);
	//layers.row (1) = Scalar (iOutputNumber);
	layers.row (1) = Scalar (iOutputNumber * 2);
	layers.row (2) = Scalar (iOutputNumber);
	oMLP.create (layers,CvANN_MLP::SIGMOID_SYM, 1, 1 );

	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 400;
	criteria.epsilon = 0.0000001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS ;;
	params.train_method = CvANN_MLP_TrainParams :: BACKPROP ;
	params.bp_dw_scale = 0.1f;
	params.bp_moment_scale = 0.1f;
	params.term_crit = criteria;

	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	oMLPParamsBI->MLPParamsField(params);

}

void MLPClassifierBI::prepareDataToEval( int i, int iItemsInSet ) 
{
	mTrainingData = Mat(0,0,CV_32FC1);
	mTrainingDataLabels = Mat(0,0,CV_32FC1);
	mTestData =  Mat(0,0,CV_32FC1);
	mTestDataLabels =  Mat(0,0,CV_32FC1);
	for (map<int, Mat>::iterator it=hCompleteData.begin(); it!=hCompleteData.end(); ++it){
		Mat vDataOneType = it->second;

		ClassifierBI::createDataToEval(i,iItemsInSet, vDataOneType, it->first, hCompleteData.size());
	}



}

void MLPClassifierBI::trainBI(){

	setParams();
	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	CvANN_MLP_TrainParams oMLPParams = oMLPParamsBI->MLPParamsField();

	if(mTrainingData.type() != CV_32FC1 &&
		(mTrainingData.type() != CV_64FC1) || mTrainingData.cols != oMLP.get_layer_sizes()->data.i[0] ){
			mTrainingData.convertTo(mTrainingData, CV_32FC1);
	}


	if(mTrainingDataLabels.type() != CV_32FC1 &&
		(mTrainingDataLabels.type() != CV_64FC1) || mTrainingDataLabels.cols != oMLP.get_layer_sizes()->data.i[oMLP.get_layer_sizes()->cols - 1]){
			
			mTrainingDataLabels.convertTo(mTrainingDataLabels, CV_32FC1);
	}

	clock_t t;
	t = clock();
	cout<< "Training neural network..." << endl;
	int iIterations = oMLP.train(mTrainingData, mTrainingDataLabels, Mat (), Mat (), oMLPParams);

	cout<< "Iterations: " << iIterations << endl;

	//mTrainingData.release();
	cout<< "Finish training neural network"<< endl;;
	t = clock() - t;
	printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
}

void MLPClassifierBI::testBI(){

	for ( int i = 0; i < mTestData.rows; i ++) {
		Mat response (1, mTestDataLabels.cols, CV_32FC1 );
		Mat sample = mTestData.row(i);


		sample.convertTo(sample, CV_32FC1);
		oMLP.predict ( sample, response );

		Mat mResonseCopy;		
		response.copyTo(mResonseCopy);
		double testMaxval;
		int maxIdx[3];
		minMaxIdx(mResonseCopy, 0, &testMaxval, 0, maxIdx);
		mResonseCopy.at<float>(maxIdx[1]) = testMaxval + 0.1;
		threshold(mResonseCopy, mResonseCopy, testMaxval, 1.0, THRESH_BINARY);

		Mat mResponse = mResonseCopy.row(0);

		Mat mLabel = mTestDataLabels.row(i);

		float fError = 0.0;

		double dLabelMaxval;
		int aLabelMaxIdx[3];
		minMaxIdx(mLabel, 0, &dLabelMaxval, 0, aLabelMaxIdx);
		int dLabelMaxIdx = aLabelMaxIdx[1];

		for (int j = 0; j< mTestDataLabels.cols ; j++)
		{
			float fResponse = mResponse.at<float>(0, j);
			float fLabel = mLabel.at<float>(0, j);

			fError += abs(fResponse - fLabel);

			double dConfusionValue =(mConfusiobMatrix.at<double>(dLabelMaxIdx, j)+ (int) fResponse);
			mConfusiobMatrix.at<double>(dLabelMaxIdx, j) = dConfusionValue;

		}

		oClassResults.AccumErr(oClassResults.AccumErr() + fError);

		if(fError < 1){
			oClassResults.TruePositive(oClassResults.TruePositive() + 1);
		}
		else{

			oClassResults.FalsePositive(oClassResults.FalsePositive() + 1);
		}
	}

}

CvANN_MLP MLPClassifierBI::MLPObject() const { return oMLP; }

void MLPClassifierBI::MLPObject(CvANN_MLP val) { oMLP = val; }

string  MLPClassifierBI::toString(){
	string sResult =  "MLP Classifier";
	return sResult;
}



