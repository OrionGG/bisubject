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

	double Efficiency() const { return (double)iTruePositive/(iTruePositive+iFalsePositive); }

	double Accurancy() const { return 1 - ((double)iFalsePositive/(iTruePositive+iFalsePositive)); }

	float AccumErr() const { return fAccumErr; }
	void AccumErr(float val) { fAccumErr = val; }

protected:
	int iTruePositive;
	int iFalsePositive;
	float fAccumErr;
};

#endif