#pragma once
#ifndef CLASSRESULTS_H
#define CLASSRESULTS_H
#include <iostream>

class ClassResults{
public:

	int TruePositive() const { return iTruePositive; }
	void TruePositive(int val) { iTruePositive = val; }

	int FalsePositive() const { return iFalsePositive; }
	void FalsePositive(int val) { iFalsePositive = val; }

	double Efficiency() const { return iTruePositive/(iTruePositive+iFalsePositive); }

	double Accurancy() const { return 1 - (iFalsePositive/(iTruePositive+iFalsePositive)); }

protected:
	int iTruePositive;
	int iFalsePositive;
};

#endif