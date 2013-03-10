#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>


#include "Global.h"
#include "ClassifierBI.h"
#include "IClassParams.h"
#include "MLPClassifierBI.h"
#include "MLPParamsBI.h"


using namespace cv;
using namespace std;
using namespace boost::filesystem;

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
	// Number of samples:
	size_t n = src.size();
	// Return empty matrix if no matrices given:
	if(n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src[0].total();
	// Create resulting data matrix:
	Mat data(n, d, rtype);

	for (int file_num = 0; file_num < n; file_num++)
	{
		Mat img_mat = src.at(file_num);
		if(img_mat.total() != d){
			resize(img_mat, img_mat, src[0].size() );
		}
		int ii = 0; // Current column in training_mat
		for (int i = 0; i<img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				data.at<float>(file_num,ii++) = img_mat.at<uchar>(i,j);
			}
		}
	}

	return data;

}


int scanFiles( const path & directory, string filterFileType, map<int, Mat> &hCompleteData, int &iMinDataPerLabel)
{   
	int fileCount=0;

	if( exists( directory ) )
	{
		directory_iterator end ;

		int iLabel = 0;

		for( directory_iterator iterDir(directory) ; iterDir != end ; ++iterDir ){
			string sSubDirectory = iterDir->path().string();
			if( exists( sSubDirectory ) )
			{
				vector<Mat> vImages;

				directory_iterator end ;
				for( directory_iterator iter(sSubDirectory) ; iter != end ; ++iter ){
					if ( (is_regular_file( *iter ) && iter->path().extension().string() == filterFileType))
					{                               
						cout << iter->path().filename().string() << endl ; //this is the one of scanned file names

						string sImageFile = iter->path().string();

						Mat mImage = imread(sImageFile );

						vImages.push_back(mImage);

						fileCount++;
					}
				}
				
				int iDataPerLabel = vImages.size();
				if (iDataPerLabel < iMinDataPerLabel)
				{
					iMinDataPerLabel =iDataPerLabel;
				}
				Mat data = asRowMatrix(vImages, CV_32FC1);
				hCompleteData.insert(pair<int,Mat>(iLabel, data));
				stringstream  sImageData;
				sImageData << ".//DataImages//" << iLabel << "data.png";
				//sImageData << "data.png";

				imwrite(sImageData.str(), data);
			}
			iLabel++;
		} 
	}
	return fileCount;
}


int main(int argc, char* argv[])
{ // lets just check the version first
	map<int, Mat> hCompleteData;

	string sImagesDataDirectory = ".//DataImages//";


	int iMinDataPerLabel = numeric_limits<int>::max();
	if(!exists( sImagesDataDirectory ) || boost::filesystem::is_empty(sImagesDataDirectory)){
			string filterFileType = ".pgm";
		int iCompleteDataCount = scanFiles("D:\\Master Vision Artificial\\BI\\Practices\\src\\Prac3\\CroppedYale", filterFileType, hCompleteData, iMinDataPerLabel);
	}
	else{	
		string filterFileType = ".png";

		directory_iterator end ;
		int iLabel = 0;
		for( directory_iterator iter(sImagesDataDirectory) ; iter != end ; ++iter ){
			if ( (is_regular_file( *iter ) && iter->path().extension().string() == filterFileType))
			{     

				string sImageFile = iter->path().string();
				Mat data = imread(sImageFile);

				int iDataPerLabel = data.rows;
				if (iDataPerLabel < iMinDataPerLabel)
				{
					iMinDataPerLabel =iDataPerLabel;

				}

				hCompleteData.insert(pair<int,Mat>(iLabel, data));
				cout << iter->path().filename().string() << endl ; //this is the one of scanned file names
				iLabel++;
			}
		}
	}


	MLPParamsBI oMLPParamsBI = MLPParamsBI();
	ClassifierBI* oClassifierBI = new MLPClassifierBI(&oMLPParamsBI);

	oClassifierBI->CompleteData(hCompleteData, iMinDataPerLabel);
	oClassifierBI->eval();
	
	delete oClassifierBI;

	return 0;
}