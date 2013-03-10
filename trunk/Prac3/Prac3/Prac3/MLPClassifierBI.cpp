#include "MLPClassifierBI.h"

MLPClassifierBI::MLPClassifierBI(MLPParamsBI* oMLPParamsBIP, int iInputNumberP, int iOutputNumberP)
	:ClassifierBI(dynamic_cast<IClassParams*>(oMLPParamsBIP))
{
	iInputNumber = iInputNumberP;
	iOutputNumber =iOutputNumberP;

	oClassResults.TruePositive(0);
	oClassResults.FalsePositive(0);
}

MLPClassifierBI::~MLPClassifierBI(){

}

void MLPClassifierBI::setParams(){
	oMLP = CvANN_MLP();
	Mat layers = Mat (2, 1, CV_32SC1 );
	//Mat layers = Mat (4, 1, CV_32SC1 );
	layers.row (0) = Scalar (iInputNumber);
	layers.row (1) = Scalar (iOutputNumber);
	//layers.row (1) = Scalar (10);
	//layers.row (2) = Scalar (15);
	//layers.row (3) = Scalar (1);
	oMLP.create (layers );

	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;	
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams :: BACKPROP;
	params.bp_dw_scale = 0.01f;
	params.bp_moment_scale = 0.01f;
	params.term_crit = criteria;

	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	oMLPParamsBI->MLPParamsField(params);

}



void MLPClassifierBI::trainBI(){

	setParams();
	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	CvANN_MLP_TrainParams oMLPParams = oMLPParamsBI->MLPParamsField();

	if(mTrainingData.type() != CV_32FC1 &&
		(mTrainingData.type() != CV_64FC1) || mTrainingData.cols != oMLP.get_layer_sizes()->data.i[0] ){
			cout<< "ERROR: input training data" << endl;
			return;
	}


	if(mTrainingDataLabels.type() != CV_32FC1 &&
		(mTrainingDataLabels.type() != CV_64FC1) || mTrainingDataLabels.cols != oMLP.get_layer_sizes()->data.i[oMLP.get_layer_sizes()->cols - 1]){

			cout<< "ERROR: output training data" << endl;
			return;
	}

	clock_t t;
	t = clock();
	cout<< "Training neural network..." << endl;
	oMLP.train(mTrainingData, mTrainingDataLabels, Mat (), Mat (), oMLPParams );
	cout<< "Finish training neural network"<< endl;;
	t = clock() - t;
	printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
}

void MLPClassifierBI::testBI(){


	for ( int i = 0; i < mTestData.rows; i ++) {
		Mat response (1, 1, CV_32FC1 );
		Mat sample = mTestData.row(i);
		oMLP.predict ( sample, response );

		float fResponse = response.at <float >(0,0)*38;

		float fLabel = mTestDataLabels.at<float>(i, 0);

		if(floor(fLabel)== floor(fResponse)){
			oClassResults.TruePositive(oClassResults.TruePositive()+1);
		}
		else{

			oClassResults.FalsePositive(oClassResults.FalsePositive()+1);
		}
	}

}

CvANN_MLP MLPClassifierBI::MLPObject() const { return oMLP; }

void MLPClassifierBI::MLPObject(CvANN_MLP val) { oMLP = val; }

string  MLPClassifierBI::toString(){
	string sResult =  "MLP Classifier";
	return sResult;
}



