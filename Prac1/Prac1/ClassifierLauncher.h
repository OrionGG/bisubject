#pragma once
#ifndef CLASSIFIERLAUNCHER_H
#define CLASSIFIERLAUNCHER_H
#include "Global.h"
#include "ClassifierBI.h"

using namespace std;

class ClassifierLauncher{
public:
	ClassifierLauncher();
	~ClassifierLauncher();

	bool setTrainingData();
	//bool setTestData();

	bool addClassifier(ClassifierBI* oClassifierBI);
	bool startClassification();

private:
	Mat mTrainingData;
	//Mat mTestData;
	vector<ClassifierBI*> vClassifierBI;

};
#endif