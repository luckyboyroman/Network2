#include "NetWork.h"

double NetWork::sigm_pro(double x)
{
	if (fabs(x) < 1e-9 || fabs(x - 1) < 1e-9) return 0;
	double res = x * (1 - x);
	return res;
}
void NetWork::SetLayers(int n, int* size) {
	srand(time(NULL));
	this->n = n;
	neurons = new neuron * [n];
	this->size = new int[n];
	weights = new double** [n - 1];
	for (int i = 0; i < n; ++i) {
		neurons[i] = new neuron[size[i]];
		this->size[i] = size[i];
		if (i < n - 1) {
			weights[i] = new double* [size[i]];
			for (int j = 0; j < size[i]; ++j) {
				weights[i][j] = new double[size[i + 1]];
				for (int k = 0; k < size[i + 1]; ++k) {
					weights[i][j][k] = (rand() % 60) / (90.0);
				}
			}
		}
	}
}
void NetWork::ShowWeights() {
	cout << "Shows weights" << endl;
	for (int i = 0; i < n - 1; ++i) {
		cout << "Layer:" << i << endl;
		for (int j = 0; j < size[i]; ++j) {
			for (int k = 0; k < size[i + 1]; ++k) {
				cout << "weights[" << i << "]" << "[" << j << "]" << "[" << k << "]\t" << weights[i][j][k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}
void NetWork::SaveWeights() {
	ofstream f_out;
	f_out.open("Weights.txt");
	if (!f_out.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < size[i]; ++j) {
			for (int k = 0; k < size[i + 1]; ++k) {
				f_out << weights[i][j][k] << " ";
			}
		}
	}
	f_out.close();
}
void NetWork::Show() {
	cout << "Shows neurons.values" << endl;
	for (int i = 0; i < n; i++) {
		cout << "Sloi" << i << endl;
		for (int j = 0; j < size[i]; j++) {
			cout << "n[" << i << "]" << "[" << j << "]" << " " << neurons[i][j].value << endl;
		}
		cout << endl;
	}
}
void NetWork::SetInput(const vector<double>& values) {
	for (int i = 0; i < size[0]; i++) {
		neurons[0][i].value = values[i];
	}
}
double NetWork::forward_feed() {
	for (int k = 1; k < n; ++k) {
		for (int i = 0; i < size[k]; ++i) {
			double sum = 0.0;
			for (int j = 0; j < size[k - 1]; ++j) {
				sum += neurons[k - 1][j].value * weights[k - 1][j][i];
			}
			neurons[k][i].value = sum; //+ShiftWeights[k][i]
			neurons[k][i].act();
		}
	}
	return neurons[n - 1][0].value;
}
void NetWork::BackPropogation(double expect) {
	neurons[n - 1][0].error = expect - neurons[n - 1][0].value;
	for (int i = n - 2; i > 0; --i) {
		for (int j = 0; j < size[i]; ++j) {
			double error = 0.0;
			for (int k = 0; k < size[i + 1]; ++k) {
				error += neurons[i + 1][k].error * weights[i][j][k];
			}
			neurons[i][j].error = error;
		}
	}
}
void NetWork::WeightsUpdater(double lr) {
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < size[i]; ++j) {
			for (int k = 0; k < size[i + 1]; ++k) {
				weights[i][j][k] += neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value * lr;
			}
		}
	}

}
void NetWork::ReadWeights() {
	fstream f_in;
	f_in.open("Weights.txt");
	if (!f_in.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < size[i]; ++j) {
			for (int k = 0; k < size[i + 1]; ++k) {
				f_in >> weights[i][j][k];
			}
		}
	}
	f_in.close();
}