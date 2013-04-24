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
#include "PCAMLPClassifierBI.h"
#include "EigenfacesParamsBI.h"
#include "EigenfacesClassifierBI.h"


using namespace cv;
using namespace std;
using namespace boost::filesystem;

int HEIGHT = 30;
int WIDTH = 30;

const int iNumComponents = 100;

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
	// Now copy data:
	for(int i = 0; i < n; i++) {
		//
		if(src[i].empty()) {
			string error_message = format("Image number %d was empty, please check your input data.", i);
			CV_Error(CV_StsBadArg, error_message);
		}

		// Get a hold of the current row:
		Mat xi = data.row(i);
		// Make reshape happy by cloning for non-continuous matrices:


		Mat img_mat = src[i];	
		if(img_mat.total() != d){
			resize(img_mat, img_mat, src[0].size() );
		}

		if(img_mat.isContinuous()) {
			img_mat.reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		} else {
			img_mat.clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}


		data.row(i) = xi ;
	}
	return data;

}




int scanFiles( const path & directory, string filterFileType, map<int, Mat> &hCompleteData, int &iMinDataPerLabel, int &iInputNumber)
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

						Mat mImage = imread(sImageFile, CV_8UC1);	
						/*cvtColor(mImage, mImage, CV_RGB2GRAY);*/
						Mat mImageResize(HEIGHT, WIDTH, CV_8UC1);
						Size s(mImageResize.rows, mImageResize.cols);
						//Size mImageSize = mImage.size();
						//size_t ss = 100;//{100, 100};
						//mImage.resize(ss);

						resize(mImage, mImageResize, s);
							//, (float) s.height/mImage.cols, (float) s.width/mImage.rows, 1);
						vImages.push_back(mImageResize);

						fileCount++;
					}
				}
				
				int iDataPerLabel = vImages.size();
				if (iDataPerLabel < iMinDataPerLabel)
				{
					iMinDataPerLabel =iDataPerLabel;
				}
				Mat data = asRowMatrix(vImages, vImages[0].type());
				iInputNumber = data.cols;
				hCompleteData.insert(pair<int,Mat>(iLabel, data));
				stringstream  sImageData;
				sImageData << ".//DataImages//" << iLabel << "data.png";
				//sImageData << "data.png";

				imwrite(sImageData.str(), data);
			}
			iLabel++;
		} 
	}
	else{
		cout << "ERROR: Folder " << directory.string() << " does not exist." << endl;
	}
	return fileCount;
}

void LoadImagesData( string sImagesDataDirectory, string filterFileType, int &iInputNumber, int &iMinDataPerLabel, map<int, Mat> &hCompleteData ) 
{
	directory_iterator end ;
	int iLabel = 0;
	for( directory_iterator iter(sImagesDataDirectory) ; iter != end ; ++iter ){
		if ( (is_regular_file( *iter ) && iter->path().extension().string() == filterFileType))
		{     

			string sImageFile = iter->path().string();
			Mat data = imread(sImageFile, CV_BGR2GRAY);
			//data.convertTo(data, CV_32FC1);
			iInputNumber = data.cols;
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


int main(int argc, char* argv[])
{ // lets just check the version first
	map<int, Mat> hCompleteData;

	string sOriginalImagesDataDirectory = ".//ExtendedYaleBPNG//";

	string sImagesDataDirectory = ".//DataImages//";


	int iMinDataPerLabel = numeric_limits<int>::max();
	int iInputNumber;
	if(!exists( sImagesDataDirectory ) || boost::filesystem::is_empty(sImagesDataDirectory)){
		//string filterFileType = ".pgm";
		string filterFileType = ".png";
		int iCompleteDataCount = scanFiles(sOriginalImagesDataDirectory, filterFileType, 
			hCompleteData, iMinDataPerLabel, iInputNumber);
		//int iCompleteDataCount = scanFiles("D:\\Master Vision Artificial\\BI\\Practices\\src\\Prac3\\Colors", filterFileType, 
		//	hCompleteData, iMinDataPerLabel, iInputNumber);
		if(iCompleteDataCount == 0){
			Sleep(10000);
			return 0;
		}
	}
	else{	
		string filterFileType = ".png";

		LoadImagesData(sImagesDataDirectory, filterFileType, iInputNumber, iMinDataPerLabel, hCompleteData);

	}

	MLPParamsBI oMLPParamsBI = MLPParamsBI();
	ClassifierBI* oClassifierBI = new MLPClassifierBI(&oMLPParamsBI, iInputNumber,hCompleteData.size());

	oClassifierBI->CompleteData(hCompleteData, iMinDataPerLabel);
	oClassifierBI->eval();

//	float accurancy = oClassifierBI->getClassResults().Accurancy();
//	float efficiency = oClassifierBI->getClassResults().Efficiency();

	MLPParamsBI oPCAMLPParamsBI = MLPParamsBI();
	ClassifierBI* oPCAClassifierBI = new PCAMLPClassifierBI(&oPCAMLPParamsBI, iInputNumber,hCompleteData.size(), 80);

	(dynamic_cast<PCAMLPClassifierBI*> (oPCAClassifierBI))->CompleteData(hCompleteData, iMinDataPerLabel);
	oPCAClassifierBI->eval();

	//EigenfacesParamsBI oEigenfacesParamsBI = EigenfacesParamsBI();
	//ClassifierBI* oEigenfacesClassifierBI = new EigenfacesClassifierBI(&oEigenfacesParamsBI, hCompleteData.size());

	//oEigenfacesClassifierBI->CompleteData(hCompleteData, iMinDataPerLabel);
	//oEigenfacesClassifierBI->eval();

	
	//delete oClassifierBI;
	delete oPCAClassifierBI;
	//delete oEigenfacesClassifierBI;

	Sleep(10000);
	return 0;
}