#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Global.h"

using namespace cv;
using namespace std;



int main(int argc, char *argv[])
{

    Mat dilated, eroded;
	
    Mat img = Global::openImage("images.jpg", CV_8UC1);
	threshold(img, img,0, 255, THRESH_OTSU|THRESH_BINARY);
	imshow("Image2", img);
    

	Size oSize(3,3);
	float PI = std::atan(1.0f) * 4.0f;
	Mat gaborKernel = getGaborKernel(oSize, 3.0, -PI/4, PI, 10.0, PI*0.5, CV_64F);
	filter2D(img, img, -1, gaborKernel);

	imshow("Image", img);
	waitKey(0);
}
