#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


int main(int argc, char *argv[])
{

    cv::Mat dilated, eroded;
    cv::Mat img = imread("1_2.jpg", CV_BGR2GRAY);
    threshold(img, img,0, 255, THRESH_OTSU|THRESH_BINARY);

    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do
    {

      cv::erode(img, eroded, element);
      cv::dilate(eroded, temp, element); // temp = open(img)
      cv::subtract(img, temp, temp);

      cv::bitwise_or(skel, temp, skel);
      eroded.copyTo(img);


      int icountNonZero = cv::countNonZero(img) ;

      done = (icountNonZero == 0);
    } while (!done);



    imshow("skel",skel);
    waitKey(0);
}
