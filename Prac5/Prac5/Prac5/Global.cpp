#include "Global.h"

using namespace std;

const char* ERROROPENFILE = "The file can not be open: ";

Mat Global::openImage(const char*  sImageFile){			
	
	Mat src;	
	
	src = imread(sImageFile);
	
	if(src.empty()){
		cout <<ERROROPENFILE << sImageFile << endl;
	}

	return src;
}

Mat Global::openImage(const char*  sImageFile, int flags){			
	
	Mat src;	
	
	src = imread(sImageFile, flags);	
	if(src.empty()){
		cout <<ERROROPENFILE << sImageFile << endl;
	}
	
	return src;
}
