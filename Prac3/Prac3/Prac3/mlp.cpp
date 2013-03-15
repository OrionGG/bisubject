#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace std ;
bool plotSupportVectors = true ;

int size =200;
int eq =0;
// accuracy
float evaluate (cv::Mat & predicted , cv::Mat& actual ) {
	assert ( predicted . rows == actual . rows );
	int t = 0;
	int f = 0;
	for ( int i = 0; i < actual . rows ; i ++) {
		float p = predicted .at <float >(i ,0) ;
		float a = actual .at <float >(i ,0) ;
		if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 && a <= 0.0) ) {
			t ++;
		} else {
			f ++;
		}
	}
	return (t * 1.0) / (t + f);
}
// plot data and class
void plot_binary (cv::Mat & data , cv::Mat& classes , string name ) {
	cv::Mat plot (size , size , CV_8UC3 );
	plot . setTo (cv::Scalar (255.0 ,255.0 ,255.0) );
	for ( int i = 0; i < data . rows ; i ++) {
		float x = data .at <float >(i ,0) * size ;
		float y = data .at <float >(i ,1) * size ;
		if( classes .at <float >(i, 0) > 0) {
			cv::circle (plot , Point (x,y), 2, CV_RGB (255 ,0 ,0) ,1);
		} else {
			cv::circle (plot , Point (x,y), 2, CV_RGB (0 ,255 ,0) ,1);
		}
	}
	cv::imshow (name , plot );
}
// function to learn
int f( float x, float y, int equation ) {
	switch ( equation ) {
	case 0:
		return y > sin (x *10) ? -1 : 1;
		break ;
	case 1:
		return y > cos (x * 10) ? -1 : 1;
		break ;
	case 2:
		return y > 2*x ? -1 : 1;
		break ;
	case 3:
		return y > tan (x *10) ? -1 : 1;
		break ;
	default :
		return y > cos (x *10) ? -1 : 1;
	}
}
// label data with equation
cv::Mat labelData (cv::Mat points , int equation ) {
	cv::Mat labels ( points .rows , 1, CV_32FC1 );
	for ( int i = 0; i < points . rows ; i ++) {
		float x = points.at <float >(i ,0) ;
		float y = points.at <float >(i ,1) ;
		labels.at <float >(i, 0) = f(x, y, equation );
	}
	return labels ;
}

void mlp (cv::Mat & trainingData , cv::Mat& trainingClasses , cv::Mat & testData , cv::Mat &
	testClasses ) {
		cv::Mat layers = cv::Mat (4, 1, CV_32SC1 );
		layers . row (0) = cv::Scalar (4) ;
		layers . row (1) = cv::Scalar (10) ;
		layers . row (2) = cv::Scalar (15) ;
		layers . row (3) = cv::Scalar (4) ;
		CvANN_MLP mlp ;
		CvANN_MLP_TrainParams params ;
		CvTermCriteria criteria ;
		criteria . max_iter = 1000;
		criteria . epsilon = 0.000001f;
		criteria . type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS ;
		params . train_method = CvANN_MLP_TrainParams :: BACKPROP ;
		params . bp_dw_scale = 0.05f;
		params . bp_moment_scale = 0.05f;
		params . term_crit = criteria ;
		mlp . create ( layers );
		// train
		int iter = mlp . train ( trainingData , trainingClasses , cv::Mat () , cv::Mat () , params );
		cv::Mat response (1, 4, CV_32FC1 );
		cv::Mat predicted ( testClasses .rows , 1, CV_32F );
		for ( int i = 0; i < testData . rows ; i ++) {
			cv::Mat response (1, 4, CV_32FC1 );
			cv::Mat sample = testData .row(i);
			mlp . predict ( sample , response );
			predicted .at <float >(i ,0) = response .at <float >(0 ,0) ;
		}
		cout << " Accuracy_ {MLP} = " << evaluate ( predicted , testClasses ) << endl ;
		plot_binary ( testData , predicted , " Predictions Backpropagation ");
}

int _main () {
	int numTrainingPoints =4;
	int numTestPoints =1;
	//cv::Mat trainingData ( numTrainingPoints , 2, CV_32FC1 );
	//cv::Mat testData ( numTestPoints , 2, CV_32FC1 );
	//cv::randu ( trainingData ,0 ,1);
	//cv::randu ( testData ,0 ,1);
	//cv::Mat trainingClasses = labelData ( trainingData , eq);
	//cv::Mat testClasses = labelData ( testData , eq);
	float vtrainingData[4][4] = {{0,0,0,0}, {64, 64,64,64}, {200,200,200,200}, {255, 255,255,255}};
	cv::Mat trainingData = Mat( numTrainingPoints , 4, CV_32FC1,&vtrainingData);

	float vtrainingClasses[4][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0},{0,0,0,1}};
	cv::Mat trainingClasses = Mat( numTrainingPoints , 4, CV_32FC1,&vtrainingClasses);

	float vtestData[1][4] ={{200,200,200,200}};
	cv::Mat testData = Mat( numTestPoints ,4, CV_32FC1,&vtestData);

	float vtestClasses[1][4] = { {0,0,1,0}};
	cv::Mat testClasses = Mat( numTestPoints , 4, CV_32FC1,&vtestClasses);

	plot_binary ( trainingData , trainingClasses , " Training Data ");
	plot_binary ( testData , testClasses , " Test Data ");
	//svm ( trainingData , trainingClasses , testData , testClasses );
	mlp ( trainingData , trainingClasses , testData , testClasses );
	//knn ( trainingData , trainingClasses , testData , testClasses , 3);
	//bayes ( trainingData , trainingClasses , testData , testClasses );
	//decisiontree ( trainingData , trainingClasses , testData , testClasses );
	cv::waitKey ();
	return 0;
}