#include "ClassifierLauncher.h"

ClassifierLauncher::ClassifierLauncher()
{

}
ClassifierLauncher::~ClassifierLauncher(){

}

bool ClassifierLauncher::setTrainingData()
{

	return true;
}
//bool setTestData();

bool ClassifierLauncher::addClassifier(ClassifierBI* oClassifierBI)
{
	vClassifierBI.push_back(oClassifierBI);
	return true;
}
bool ClassifierLauncher::startClassification()
{
	for (int i= 0; i< vClassifierBI.size();i++)
	{
		vClassifierBI[i]->eval(PercCrossFold);
	}

	return true;
}