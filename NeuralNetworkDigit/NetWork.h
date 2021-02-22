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
	void SetLayers(int n, int* size);
	void SaveWeights();
	void ReadWeights();
	void Show();
	void ShowWeights();
	void SetInput(const vector<double>& values);
	double forward_feed();
	void BackPropogation(double expect);
	void WeightsUpdater(double lr);
};