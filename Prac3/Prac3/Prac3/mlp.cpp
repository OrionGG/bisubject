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

const int cols =  10000;
void mlp (cv::Mat & trainingData , cv::Mat& trainingClasses , cv::Mat & testData , cv::Mat &
	testClasses ) {
        cv::Mat layers = cv::Mat (5, 1, CV_32SC1 );
        layers . row (0) = cv::Scalar (cols) ;
//        layers . row (1) = cv::Scalar (cols*5) ;
//        layers . row (2) = cv::Scalar (cols*5*3/2) ;
        layers . row (1) = cv::Scalar (10) ;
        layers . row (2) = cv::Scalar (15) ;
        layers . row (3) = cv::Scalar (10) ;
        //layers . row (1) = cv::Scalar (cols*5) ;
        //layers . row (2) = cv::Scalar (4) ;
        layers . row (4) = cv::Scalar (4) ;
		CvANN_MLP mlp ;
		CvANN_MLP_TrainParams params ;
		CvTermCriteria criteria ;
        criteria . max_iter = 100000;
        criteria . epsilon = 0.0000000001f;
		criteria . type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS ;
		params . train_method = CvANN_MLP_TrainParams :: BACKPROP ;
        params . bp_dw_scale = 0.001f;
        params . bp_moment_scale = 0.00001f;
		params . term_crit = criteria ;
        mlp . create ( layers );
		int iter = mlp . train ( trainingData , trainingClasses , cv::Mat () , cv::Mat () , params );
        cout << iter << endl;
        cv::Mat response (1, 5, CV_32FC1 );
        cv::Mat predicted ( testClasses .rows , 5, CV_32F );
		for ( int i = 0; i < testData . rows ; i ++) {
            cv::Mat response (1, 5, CV_32FC1 );
			cv::Mat sample = testData .row(i);
			mlp . predict ( sample , response );
            for(int j =0; j < response.cols; j++){
                float fresponse = response.at <float >(0 ,j) ;
                cout << fresponse << endl;
                predicted .at <float >(i ,0) = response .at <float >(0 ,0) ;
            }
		}
        cout << " Accuracy_ {MLP} = " << evaluate ( predicted , testClasses ) << endl ;
		plot_binary ( testData , predicted , " Predictions Backpropagation ");
}

int _main () {
    const int numTrainingPoints =4;
    const int numTestPoints =1;

	//cv::Mat trainingData ( numTrainingPoints , 2, CV_32FC1 );
	//cv::Mat testData ( numTestPoints , 2, CV_32FC1 );
	//cv::randu ( trainingData ,0 ,1);
	//cv::randu ( testData ,0 ,1);
	//cv::Mat trainingClasses = labelData ( trainingData , eq);
    //cv::Mat testClasses = labelData ( testData , eq);
    float vtrainingData[numTrainingPoints][cols];
    for(int i= 0; i < numTrainingPoints;i++){
        for(int j= 0; j < cols;j++){
            vtrainingData[i][j] = (float)(255/3)*i;
        }
    }
    cv::Mat trainingData = Mat( numTrainingPoints , cols, CV_32FC1,&vtrainingData);

    float vtrainingClasses[numTrainingPoints][numTrainingPoints] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0},{0,0,0,1}};
    cv::Mat trainingClasses = Mat( numTrainingPoints , numTrainingPoints, CV_32FC1,&vtrainingClasses);

    float vtestData[1][cols];
    for(int j= 0; j < cols;j++){
        vtestData[0][j] = (float)(255/3)*2;
    }
    cv::Mat testData = Mat( numTestPoints ,cols, CV_32FC1,&vtestData);

    float vtestClasses[1][numTrainingPoints] = { {0,0,1,0}};
    cv::Mat testClasses = Mat( numTestPoints , numTrainingPoints, CV_32FC1,&vtestClasses);

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
