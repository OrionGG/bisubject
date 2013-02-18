#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include "ClassifierLauncher.h"
#include "ClassifierBI.h"
#include "IClassParams.h"
#include "SVMClassifierBI.h"
#include "NBClassifierBI.h"
#include "SVMParamsBI.h"
#include "NBParamsBI.h"



using namespace cv;
using namespace std;

void readLine( char * &pch, double &dCoord, vector<float> &mResult, int &cols ) 
{
	while (pch != NULL &&  !isspace(*pch))
	{
		dCoord = atof(pch);
		mResult.push_back(dCoord);
		

		pch = strtok(NULL, " ");
		cols++;

	}
}

void deleteLine( vector<float> &mResult ) 
{
	mResult.pop_back();
	mResult.pop_back();
	mResult.pop_back();
	mResult.pop_back();
}

vector<float>  readFileText  (char* inputFile, 	vector<float> vPointsToClassify, int& rows, int& cols) {

	rows = 0;
	vector<float> mResult;
	string line;
	ifstream myfile (inputFile);

	if (myfile.is_open())
	{
		while ( myfile.good() )
		{
			getline (myfile,line);

			if(!line.empty()){
				if((line[0] != '-' && isdigit(line[0]))||
					(line[0] == '-' && isdigit(line[1]))){
						char * pch = strtok(&line[0]," ");
						cols = 0;
						double dCoord = 0;
						readLine(pch, dCoord, mResult, cols);


						if(find(vPointsToClassify.begin(), vPointsToClassify.end(), dCoord) == vPointsToClassify.end()){
							deleteLine(mResult);

						}
						else{
						
							rows++;
						}
				}
			}
			
		}
		myfile.close();
	}
	else{

	}

	return mResult;
}

void createClassifiers( int rows, int cols, vector<float> vData ) 
{
	Mat myMat(rows, cols, CV_32FC1, &vData[0]);

	ClassifierLauncher oClassifierLauncher = ClassifierLauncher();

	SVMParamsBI oSVMParam = SVMParamsBI();
	SVMParams params = SVMParams();
	oSVMParam.SVMParamsField(params);

	SVMClassifierBI* oSVMClassifierBI = new SVMClassifierBI(&oSVMParam);

	oClassifierLauncher.addClassifier(oSVMClassifierBI);

	NBParamsBI oNBParamsBI = NBParamsBI();
	NBClassifierBI* oNBClassifierBI = new NBClassifierBI(&oNBParamsBI);

	oClassifierLauncher.addClassifier(oNBClassifierBI);
	oClassifierLauncher.setCompleteData(myMat);
	//oClassifierLauncher.setTrainingData(vData);
	oClassifierLauncher.startClassification();
}



int main(int argc, char *argv[])
{
	int rows, cols = 0;

	static const float aPointsToClassify[] = {0, 2, 8};
	cout << "Selected labels: " << endl;
	int n = sizeof(aPointsToClassify)/sizeof(aPointsToClassify[0]);
	for(int i = 0; i < n;i++) cout << aPointsToClassify[i] << endl;
	cout << endl;

	vector<float> vPointsToClassify (aPointsToClassify, aPointsToClassify + sizeof(aPointsToClassify) / sizeof(aPointsToClassify[0]) );

	char* inputFile = ".\\resources\\prac1_fichPuntosFaciales.txt";
	vector<float> vData = readFileText(inputFile, vPointsToClassify, rows, cols);
	if(vData.size() == 0){
		cout << ERROROPENFILE << inputFile << endl;
	}
	else{
		createClassifiers(rows, cols, vData);

	}

	Sleep(10000);

}
