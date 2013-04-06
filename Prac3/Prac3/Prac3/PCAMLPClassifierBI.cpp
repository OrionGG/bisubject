#include "PCAMLPClassifierBI.h"

PCAMLPClassifierBI::PCAMLPClassifierBI(MLPParamsBI* oMLPParamsBIP, int iInputNumberP, int iOutputNumberP, int iNumComponentsP)
	:MLPClassifierBI(oMLPParamsBIP, iNumComponentsP, iOutputNumberP)
{
	iNumComponents = iNumComponentsP;
}

PCAMLPClassifierBI::~PCAMLPClassifierBI(){

}


void PCAMLPClassifierBI::setParams(){
	
	if(oMLP.get_layer_count() == 0){
		oMLP = CvANN_MLP();
		//Mat layers = Mat (2, 1, CV_32SC1 );
		Mat layers = Mat (3, 1, CV_32SC1 );
		layers.row (0) = Scalar (iNumComponents);
		//layers.row (1) = Scalar (iOutputNumber);
		layers.row (1) = Scalar (iOutputNumber * 2);
		layers.row (2) = Scalar (iOutputNumber);
		oMLP.create (layers,CvANN_MLP::SIGMOID_SYM, 1,1 );

	}

	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 0.0000001f;
	//criteria.type = CV_TERMCRIT_ITER ;
	criteria.type = CV_TERMCRIT_ITER ;
	params.train_method = CvANN_MLP_TrainParams :: BACKPROP ;
	params.bp_dw_scale = 0.1f;
	params.bp_moment_scale = 0.1f;
	params.term_crit = criteria;

	MLPParamsBI* oMLPParamsBI = static_cast<MLPParamsBI*>(oClassParams);
	oMLPParamsBI->MLPParamsField(params);

}

// Normalizes a given image into a value range between 0 and 255.
Mat PCAMLPClassifierBI::norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch(src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

void PCAMLPClassifierBI::prepareDataToEval( int i, int iItemsInSet ){

	mTrainingData = Mat(0,0,CV_32FC1);
	mTrainingDataLabels = Mat(0,0,CV_32FC1);
	mTestData =  Mat(0,0,CV_32FC1);
	mTestDataLabels =  Mat(0,0,CV_32FC1);


	string sOutputPCAFolder = "D:\\Master Vision Artificial\\BI\\Practices\\src\\Prac3\\Prac3\\Prac3\\PCAs\\";

	for (map<int, Mat>::iterator it=hCompleteData.begin(); it!=hCompleteData.end(); ++it){
		string sOutputPCAFolder = "D:\\Master Vision Artificial\\BI\\Practices\\src\\Prac3\\Prac3\\Prac3\\PCAs\\" + to_string(static_cast<long long>(it->first));
		Mat vDataOneType;
		if( !exists( sOutputPCAFolder ) )
		{
			// Number of components to keep for the PCA:
			//PCA oPCA(mCompleteDataSVD, Mat(), CV_PCA_DATA_AS_ROW, mCompleteDataSVD.cols);

			vDataOneType = it->second;	
			Mat mDataPCA;
			vDataOneType.copyTo(mDataPCA);
			vDataOneType = Mat(0,0,vDataOneType.type());
			
			for (int iRow=0; iRow < mDataPCA.rows; iRow++)
			{
				Mat mImage = mDataPCA.row(iRow);
				//if(mImage.type() != CV_32FC1 ||
				//	(mImage.type() != CV_64FC1)){
				//		mImage.convertTo(mImage, CV_32FC1);
				//}
				//mImage = mImage.reshape(1, 30);
				//SVD oSVD(mImage,cv::SVD::NO_UV);
				//Mat w = oSVD.w.reshape(1, 1);

				Mat mImpageProjected;
				oPCA.project(mImage, mImpageProjected);
				Mat mImpageProjectedNorm = norm_0_255(mImpageProjected);
				vDataOneType.push_back(mImpageProjectedNorm);

			}

			boost::filesystem::create_directory(sOutputPCAFolder);
			imwrite(sOutputPCAFolder + "\\PCAData.png", vDataOneType);
		}
		else{
			vDataOneType = imread(sOutputPCAFolder + "\\PCAData.png", CV_BGR2GRAY);
		}
	

		ClassifierBI::createDataToEval(i,iItemsInSet, vDataOneType, it->first, hCompleteData.size());


	}


}

void PCAMLPClassifierBI::CompleteData(map<int, Mat> val, int iMinDataPerLabelP) { 

	hCompleteData = val; 
	iMinDataPerLabel = iMinDataPerLabelP;

	for (map<int, Mat>::iterator it=hCompleteData.begin(); it!=hCompleteData.end(); ++it){

		Mat vDataOneType = it->second;	

		mCompleteData.push_back(vDataOneType);

	}


	//oPCA(mCompleteDataSVD, Mat(),  mCompleteDataSVD.cols);
	oPCA(mCompleteData, Mat(), CV_PCA_DATA_AS_ROW, iNumComponents);
	//oPCA(mCompleteData, Mat(), CV_PCA_DATA_AS_ROW, mCompleteData.cols);
	
}

