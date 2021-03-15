#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <random>
#include <fstream>
#include <iostream>
using namespace std;

struct neuron {
	double value;
	double error;
	void act() {
		value = 1 / (1 + pow(2.71828, -value));
	}
};
class NetWork
{
	int n;
	int* size;
public:
	neuron** neurons;
	double*** weights;
	double sigm_pro(double x);
	double ReLUpro(double x);
	void SetLayers(int n, int* size);
	void SaveWeights();
	void ReadWeights();
	void Show();
	void ShowWeights();
	void SetInput(double* values);
	double forward_feed();
	double forward_feed(bool flag);
	void BackPropogation(double expect);
	double ErrorCounter();
	void WeightsUpdater(double lr);
};
