// Copyright 2025 ELM-C++ Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ELM Training Command-Line Utility
// Trains an Extreme Learning Machine model on CSV data

#include "elml.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace Eigen;

// Structure to hold command-line arguments
struct Arguments {
	string trainFile;
	string modelFile;
	int nhn;
	double C;
	bool hasLabels;
	int labelColumn;
	
	Arguments() : trainFile(""), modelFile("model.elm"), nhn(0), C(1.0), 
	              hasLabels(true), labelColumn(-1) {}
};

// Print usage information
void printUsage(const char* progName) {
	cout << "ELM Training Utility" << endl;
	cout << "Usage: " << progName << " [options]" << endl;
	cout << endl;
	cout << "Required arguments:" << endl;
	cout << "  -i <file>        Input CSV training file" << endl;
	cout << endl;
	cout << "Optional arguments:" << endl;
	cout << "  -o <file>        Output model file (default: model.elm)" << endl;
	cout << "  -n <int>         Number of hidden neurons (default: features/2)" << endl;
	cout << "  -c <double>      Regularization parameter C (default: 1.0)" << endl;
	cout << "  -l <int>         Label column index (default: last column, -1)" << endl;
	cout << "  --no-header      First row is data, not header" << endl;
	cout << "  -h, --help       Show this help message" << endl;
	cout << endl;
	cout << "CSV Format:" << endl;
	cout << "  - Features in columns, samples in rows" << endl;
	cout << "  - Last column should contain class labels (or use -l to specify)" << endl;
	cout << "  - First row can be header (skipped by default)" << endl;
	cout << endl;
}

// Parse command-line arguments
bool parseArguments(int argc, char* argv[], Arguments& args) {
	bool hasHeader = true;
	
	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		
		if (arg == "-h" || arg == "--help") {
			return false;
		}
		else if (arg == "-i" && i + 1 < argc) {
			args.trainFile = argv[++i];
		}
		else if (arg == "-o" && i + 1 < argc) {
			args.modelFile = argv[++i];
		}
		else if (arg == "-n" && i + 1 < argc) {
			args.nhn = atoi(argv[++i]);
		}
		else if (arg == "-c" && i + 1 < argc) {
			args.C = atof(argv[++i]);
		}
		else if (arg == "-l" && i + 1 < argc) {
			args.labelColumn = atoi(argv[++i]);
		}
		else if (arg == "--no-header") {
			hasHeader = false;
		}
		else {
			cerr << "Unknown argument: " << arg << endl;
			return false;
		}
	}
	
	if (args.trainFile.empty()) {
		cerr << "Error: Input file is required (-i)" << endl;
		return false;
	}
	
	return true;
}

// Read CSV file and return data matrix and labels
bool readCSV(const string& filename, MatrixXd& X, VectorXd& Y, int labelColumn, bool hasHeader = true) {
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot open file " << filename << endl;
		return false;
	}
	
	vector<vector<double>> data;
	vector<double> labels;
	string line;
	bool firstLine = true;
	
	while (getline(file, line)) {
		// Skip header if present
		if (firstLine && hasHeader) {
			firstLine = false;
			continue;
		}
		firstLine = false;
		
		// Skip empty lines
		if (line.empty() || line.find_first_not_of(" \t\r\n") == string::npos) {
			continue;
		}
		
		stringstream ss(line);
		string cell;
		vector<double> row;
		int colIdx = 0;
		
		while (getline(ss, cell, ',')) {
			// Trim whitespace
			size_t start = cell.find_first_not_of(" \t\r\n");
			size_t end = cell.find_last_not_of(" \t\r\n");
			if (start != string::npos && end != string::npos) {
				cell = cell.substr(start, end - start + 1);
			}
			
			if (!cell.empty()) {
				row.push_back(atof(cell.c_str()));
			}
			colIdx++;
		}
		
		if (row.empty()) continue;
		
		// Extract label (default: last column)
		int lblCol = (labelColumn < 0) ? (row.size() - 1) : labelColumn;
		if (lblCol >= 0 && lblCol < static_cast<int>(row.size())) {
			labels.push_back(row[lblCol]);
			row.erase(row.begin() + lblCol);
		}
		
		data.push_back(row);
	}
	
	file.close();
	
	if (data.empty()) {
		cerr << "Error: No data read from file" << endl;
		return false;
	}
	
	// Convert to Eigen matrices
	int nSamples = data.size();
	int nFeatures = data[0].size();
	
	X.resize(nFeatures, nSamples);
	Y.resize(nSamples);
	
	for (int i = 0; i < nSamples; i++) {
		if (static_cast<int>(data[i].size()) != nFeatures) {
			cerr << "Error: Inconsistent number of features at row " << i << endl;
			return false;
		}
		for (int j = 0; j < nFeatures; j++) {
			X(j, i) = data[i][j];
		}
		Y(i) = labels[i];
	}
	
	cout << "Loaded " << nSamples << " samples with " << nFeatures << " features" << endl;
	
	return true;
}

// Save model to file
bool saveModel(const string& filename, const MatrixXd& inW, const MatrixXd& bias, const MatrixXd& outW) {
	ofstream file(filename, ios::binary);
	if (!file.is_open()) {
		cerr << "Error: Cannot create model file " << filename << endl;
		return false;
	}
	
	// Write dimensions
	int inW_rows = inW.rows();
	int inW_cols = inW.cols();
	int bias_rows = bias.rows();
	int outW_rows = outW.rows();
	int outW_cols = outW.cols();
	
	file.write(reinterpret_cast<const char*>(&inW_rows), sizeof(int));
	file.write(reinterpret_cast<const char*>(&inW_cols), sizeof(int));
	file.write(reinterpret_cast<const char*>(&bias_rows), sizeof(int));
	file.write(reinterpret_cast<const char*>(&outW_rows), sizeof(int));
	file.write(reinterpret_cast<const char*>(&outW_cols), sizeof(int));
	
	// Write matrix data
	file.write(reinterpret_cast<const char*>(inW.data()), inW_rows * inW_cols * sizeof(double));
	file.write(reinterpret_cast<const char*>(bias.data()), bias_rows * sizeof(double));
	file.write(reinterpret_cast<const char*>(outW.data()), outW_rows * outW_cols * sizeof(double));
	
	file.close();
	
	cout << "Model saved to " << filename << endl;
	return true;
}

int main(int argc, char* argv[]) {
	Arguments args;
	
	if (!parseArguments(argc, argv, args)) {
		printUsage(argv[0]);
		return 1;
	}
	
	// Read training data
	MatrixXd X;
	VectorXd Y;
	
	if (!readCSV(args.trainFile, X, Y, args.labelColumn)) {
		return 1;
	}
	
	// Set default number of hidden neurons if not specified
	if (args.nhn <= 0) {
		args.nhn = X.rows() / 2;
		if (args.nhn < 1) args.nhn = 1;
	}
	
	cout << "Training ELM model..." << endl;
	cout << "  Hidden neurons: " << args.nhn << endl;
	cout << "  Regularization C: " << args.C << endl;
	
	// Train model
	MatrixXd inW, bias, outW;
	
	int result = elmTrain(X.data(), X.rows(), X.cols(),
	                      Y.data(),
	                      args.nhn, args.C,
	                      inW, bias, outW);
	
	if (result != 0) {
		cerr << "Error: Training failed" << endl;
		return 1;
	}
	
	cout << "Training completed successfully!" << endl;
	cout << "  Model dimensions:" << endl;
	cout << "    Input weights: " << inW.rows() << " x " << inW.cols() << endl;
	cout << "    Bias: " << bias.rows() << " x " << bias.cols() << endl;
	cout << "    Output weights: " << outW.rows() << " x " << outW.cols() << endl;
	
	// Save model
	if (!saveModel(args.modelFile, inW, bias, outW)) {
		return 1;
	}
	
	return 0;
}

