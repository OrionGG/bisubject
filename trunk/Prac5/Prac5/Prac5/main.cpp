#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Global.h"

using namespace cv;
using namespace std;

/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;


}


int kernel_size=15;
int pos_sigma= 3;
int pos_lm = 8;
int pos_gm = 1000;
int pos_psi = 0;
cv::Mat img;
cv::Mat dest1;
cv::Mat dest2;
cv::Mat dest3;
cv::Mat dest4;
cv::Mat dest;

void Process(int , void *)
{
	double sig = pos_sigma;
	double lm = (double)CV_PI/4 * pos_lm;
	double gm = (double)pos_gm;
	double ps = (double)CV_PI/2 * pos_psi;
	Size oSize(kernel_size,kernel_size);
	Mat gaborKernel1 = getGaborKernel(oSize, sig, 0, lm, gm, ps, CV_32F);
	filter2D(img, dest1, img.type(), gaborKernel1);
	cv::imshow("Process window1", dest1);
	Mat gaborKernel2 = getGaborKernel(oSize, sig, (double)CV_PI/4, lm, gm, ps, CV_32F);
	filter2D(img, dest2, img.type(), gaborKernel2);
	cv::imshow("Process window2", dest2);
	Mat gaborKernel3 = getGaborKernel(oSize, sig, (double)CV_PI/2, lm, gm, ps, CV_32F);
	filter2D(img, dest3, img.type(), gaborKernel3);
	cv::imshow("Process window3", dest3);
	Mat gaborKernel4 = getGaborKernel(oSize, sig, (double)CV_PI/4 * 3, lm, gm, ps, CV_32F);
	filter2D(img, dest4, img.type(), gaborKernel4);
	cv::imshow("Process window4", dest4);
	dest = (dest1 + dest2 + dest3 +dest4)/4;
	cv::imshow("Process window0", dest);
	dest.convertTo(dest, CV_8UC1);
	threshold(dest, dest, 0,255, THRESH_OTSU|THRESH_BINARY);
	cv::imshow("Process window", dest);
}


int main(int argc, char *argv[])
{


	//cv::Mat src = cv::imread(".\\DB4_B\\101_1.tif");
	//if (src.empty())
	//	return -1;

	//cv::Mat bw;
	//cv::cvtColor(src, bw, CV_BGR2GRAY); 
	//equalizeHist( bw, bw );
	//cv::threshold(bw, bw, 50, 255, CV_THRESH_BINARY);
	//cv::imshow("dst", bw);
	//cv::waitKey(0);

	//thinning(bw);

	//cv::imshow("src", src);
	//cv::imshow("dst", bw);
	//cv::waitKey(0);

	//return 0;
	

	img = cv::imread(".\\DB4_B\\101_1.tif");
	cv::imshow("Src", img);
	cv::Mat src;
	cv::cvtColor(img, src, CV_BGR2GRAY);
	src.convertTo(img, CV_32F, 1.0/255, 0);
	if (!kernel_size%2)
	{
		kernel_size+=1;
	}
	cv::namedWindow("Process window", 1);
	cv::createTrackbar("Kernel", "Process window", &kernel_size, 21, Process);
	cv::createTrackbar("Sigma", "Process window", &pos_sigma, 10, Process);
	cv::createTrackbar("Lambda", "Process window", &pos_lm, 10, Process);
	cv::createTrackbar("Gamma", "Process window", &pos_gm, 2000, Process);
	cv::createTrackbar("Psi", "Process window", &pos_psi, 4, Process);
	Process(0,0);


	waitKey(0);

	return 0;

}
