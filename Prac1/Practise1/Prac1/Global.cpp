#include "Global.h"

const int PercCrossFold = 10;
const char* ERROROPENFILE = "The file can not be open: ";

Mat Global::openImage(const char*  sImageFile){			
	
	Mat src;	
	
	src = imread(sImageFile);
	
	return src;
}

Mat Global::openImage(const char*  sImageFile, int flags){			
	
	Mat src;	
	
	src = imread(sImageFile, flags);
	
	return src;
}
