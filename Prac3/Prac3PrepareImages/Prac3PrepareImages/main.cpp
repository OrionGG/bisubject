#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>




using namespace cv;
using namespace std;
using namespace boost::filesystem;

int HEIGHT = 100;
int WIDTH = 100;

const int iNumComponents = 30;

int convertToPNG( const path & directory, string filterFileType, string sOutputFolder)
{   
	int fileCount=0;

	if( exists( directory ) )
	{
		directory_iterator end ;

		int iLabel = 0;

		for( directory_iterator iterDir(directory) ; iterDir != end ; ++iterDir ){
			string sSubDirectory = iterDir->path().string();

			size_t found;
			cout << "Splitting: " << sSubDirectory << endl;
			found=sSubDirectory.find_last_of("/\\");
			cout << " folder: " << sSubDirectory.substr(0,found) << endl;
			cout << " file: " << sSubDirectory.substr(found+1) << endl;
			string sSubFolderName = sSubDirectory.substr(found+1);

			string sOutputFolderImages = sOutputFolder + "\\" + sSubFolderName;
			boost::filesystem::create_directory(sOutputFolderImages);

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
						Mat mImageResized(HEIGHT, WIDTH, CV_8UC1);
						Size s(mImageResized.rows, mImageResized.cols);


						resize(mImage, mImageResized, s);

						
						
						stringstream  sImageData;
						sImageData << sOutputFolderImages << "\\" << iter->path().filename().string() << ".png";
						//sImageData << "data.png";

						bool bWrite = imwrite(sImageData.str(), mImageResized);
					}
				}
			}
			iLabel++;
		} 
	}
	else{
		cout << "ERROR: Folder " << directory.string() << " does not exist." << endl;
	}
	return fileCount;
}



int main(int argc, char* argv[])
{ // lets just check the version first
	string sImagesDataDirectory = ".\\ExtendedYaleB";

	string sOutputFolder = ".\\ExtendedYaleBPNG\\";
	
	string filterFileType = ".pgm";

	int iCompleteDataCount = convertToPNG(sImagesDataDirectory, filterFileType, sOutputFolder);

	
	Sleep(10000);
	return 0;
}