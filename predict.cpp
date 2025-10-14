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

// ELM Prediction Command-Line Utility
// Predicts class labels using a trained ELM model

#include "elml.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace Eigen;

// Structure to hold command-line arguments
struct Arguments {
	string testFile;
	string modelFile;
	string outputFile;
	bool hasHeader;
	bool outputScores;
	int labelColumn;
	
	Arguments() : testFile(""), modelFile("model.elm"), outputFile("predictions.csv"),
	              hasHeader(true), outputScores(false), labelColumn(-2) {}
};

// Print usage information
void printUsage(const char* progName) {
	cout << "ELM Prediction Utility" << endl;
	cout << "Usage: " << progName << " [options]" << endl;
	cout << endl;
	cout << "Required arguments:" << endl;
	cout << "  -i <file>        Input CSV test file" << endl;
	cout << "  -m <file>        Input model file (trained ELM model)" << endl;
	cout << endl;
	cout << "Optional arguments:" << endl;
	cout << "  -o <file>        Output predictions file (default: predictions.csv)" << endl;
	cout << "  -l <int>         Label column to skip: -1=last col, >=0=specific col, -2=none (default: -2)" << endl;
	cout << "  --no-header      First row is data, not header" << endl;
	cout << "  --scores         Output all class scores (not just predicted label)" << endl;
	cout << "  -h, --help       Show this help message" << endl;
	cout << endl;
	cout << "CSV Format:" << endl;
	cout << "  - Features in columns, samples in rows" << endl;
	cout << "  - Should have same number of features as training data" << endl;
	cout << "  - Use -l to skip label column if present (e.g., -l -1 to skip last column)" << endl;
	cout << "  - First row can be header (skipped by default)" << endl;
	cout << endl;
}

// Parse command-line arguments
bool parseArguments(int argc, char* argv[], Arguments& args) {
	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		
		if (arg == "-h" || arg == "--help") {
			return false;
		}
		else if (arg == "-i" && i + 1 < argc) {
			args.testFile = argv[++i];
		}
		else if (arg == "-m" && i + 1 < argc) {
			args.modelFile = argv[++i];
		}
		else if (arg == "-o" && i + 1 < argc) {
			args.outputFile = argv[++i];
		}
		else if (arg == "-l" && i + 1 < argc) {
			args.labelColumn = atoi(argv[++i]);
		}
		else if (arg == "--no-header") {
			args.hasHeader = false;
		}
		else if (arg == "--scores") {
			args.outputScores = true;
		}
		else {
			cerr << "Unknown argument: " << arg << endl;
			return false;
		}
	}
	
	if (args.testFile.empty()) {
		cerr << "Error: Input test file is required (-i)" << endl;
		return false;
	}
	
	if (args.modelFile.empty()) {
		cerr << "Error: Model file is required (-m)" << endl;
		return false;
	}
	
	return true;
}

// Read CSV file and return data matrix
// labelColumn: -2 = no label column, -1 = last column, >= 0 = specific column index
bool readCSV(const string& filename, MatrixXd& X, int labelColumn = -2, bool hasHeader = true) {
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot open file " << filename << endl;
		return false;
	}
	
	vector<vector<double>> data;
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
		}
		
		if (!row.empty()) {
			// Remove label column if specified
			if (labelColumn >= -1) {
				int lblCol = (labelColumn < 0) ? (row.size() - 1) : labelColumn;
				if (lblCol >= 0 && lblCol < static_cast<int>(row.size())) {
					row.erase(row.begin() + lblCol);
				}
			}
			
			if (!row.empty()) {
				data.push_back(row);
			}
		}
	}
	
	file.close();
	
	if (data.empty()) {
		cerr << "Error: No data read from file" << endl;
		return false;
	}
	
	// Convert to Eigen matrix
	int nSamples = data.size();
	int nFeatures = data[0].size();
	
	X.resize(nFeatures, nSamples);
	
	for (int i = 0; i < nSamples; i++) {
		if (static_cast<int>(data[i].size()) != nFeatures) {
			cerr << "Error: Inconsistent number of features at row " << i << endl;
			return false;
		}
		for (int j = 0; j < nFeatures; j++) {
			X(j, i) = data[i][j];
		}
	}
	
	cout << "Loaded " << nSamples << " samples with " << nFeatures << " features" << endl;
	
	return true;
}

// Load model from file
bool loadModel(const string& filename, MatrixXd& inW, MatrixXd& bias, MatrixXd& outW) {
	ifstream file(filename, ios::binary);
	if (!file.is_open()) {
		cerr << "Error: Cannot open model file " << filename << endl;
		return false;
	}
	
	// Read dimensions
	int inW_rows, inW_cols, bias_rows, outW_rows, outW_cols;
	
	file.read(reinterpret_cast<char*>(&inW_rows), sizeof(int));
	file.read(reinterpret_cast<char*>(&inW_cols), sizeof(int));
	file.read(reinterpret_cast<char*>(&bias_rows), sizeof(int));
	file.read(reinterpret_cast<char*>(&outW_rows), sizeof(int));
	file.read(reinterpret_cast<char*>(&outW_cols), sizeof(int));
	
	// Validate dimensions
	if (inW_rows <= 0 || inW_cols <= 0 || bias_rows <= 0 || outW_rows <= 0 || outW_cols <= 0) {
		cerr << "Error: Invalid model dimensions" << endl;
		return false;
	}
	
	if (inW_rows != bias_rows || inW_rows != outW_rows) {
		cerr << "Error: Inconsistent model dimensions" << endl;
		return false;
	}
	
	// Resize matrices
	inW.resize(inW_rows, inW_cols);
	bias.resize(bias_rows, 1);
	outW.resize(outW_rows, outW_cols);
	
	// Read matrix data
	file.read(reinterpret_cast<char*>(inW.data()), inW_rows * inW_cols * sizeof(double));
	file.read(reinterpret_cast<char*>(bias.data()), bias_rows * sizeof(double));
	file.read(reinterpret_cast<char*>(outW.data()), outW_rows * outW_cols * sizeof(double));
	
	file.close();
	
	cout << "Model loaded from " << filename << endl;
	cout << "  Input weights: " << inW.rows() << " x " << inW.cols() << endl;
	cout << "  Bias: " << bias.rows() << " x " << bias.cols() << endl;
	cout << "  Output weights: " << outW.rows() << " x " << outW.cols() << endl;
	
	return true;
}

// Save predictions to CSV file
bool savePredictions(const string& filename, const MatrixXd& scores, bool outputScores) {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot create output file " << filename << endl;
		return false;
	}
	
	int nSamples = scores.cols();
	int nClasses = scores.rows();
	
	if (outputScores) {
		// Output all scores for each class
		file << "sample";
		for (int c = 0; c < nClasses; c++) {
			file << ",class_" << (c + 1) << "_score";
		}
		file << ",predicted_class" << endl;
		
		for (int i = 0; i < nSamples; i++) {
			file << i;
			
			// Find max score
			int maxIdx = 0;
			double maxScore = scores(0, i);
			for (int c = 0; c < nClasses; c++) {
				file << "," << scores(c, i);
				if (scores(c, i) > maxScore) {
					maxScore = scores(c, i);
					maxIdx = c;
				}
			}
			file << "," << (maxIdx + 1) << endl;
		}
	}
	else {
		// Output only predicted class label
		file << "predicted_class" << endl;
		
		for (int i = 0; i < nSamples; i++) {
			// Find max score
			int maxIdx = 0;
			double maxScore = scores(0, i);
			for (int c = 1; c < nClasses; c++) {
				if (scores(c, i) > maxScore) {
					maxScore = scores(c, i);
					maxIdx = c;
				}
			}
			file << (maxIdx + 1) << endl;
		}
	}
	
	file.close();
	cout << "Predictions saved to " << filename << endl;
	
	return true;
}

int main(int argc, char* argv[]) {
	Arguments args;
	
	if (!parseArguments(argc, argv, args)) {
		printUsage(argv[0]);
		return 1;
	}
	
	// Load model
	MatrixXd inW, bias, outW;
	if (!loadModel(args.modelFile, inW, bias, outW)) {
		return 1;
	}
	
	// Read test data
	MatrixXd X;
	if (!readCSV(args.testFile, X, args.labelColumn, args.hasHeader)) {
		return 1;
	}
	
	// Validate dimensions
	if (X.rows() != inW.cols()) {
		cerr << "Error: Feature dimension mismatch. Model expects " << inW.cols() 
		     << " features, but test data has " << X.rows() << " features." << endl;
		return 1;
	}
	
	cout << "Predicting..." << endl;
	
	// Predict
	MatrixXd scores;
	int result = elmPredict(X.data(), X.rows(), X.cols(),
	                        scores, inW, bias, outW);
	
	if (result != 0) {
		cerr << "Error: Prediction failed" << endl;
		return 1;
	}
	
	cout << "Prediction completed!" << endl;
	cout << "  " << scores.cols() << " samples classified into " 
	     << scores.rows() << " classes" << endl;
	
	// Save predictions
	if (!savePredictions(args.outputFile, scores, args.outputScores)) {
		return 1;
	}
	
	return 0;
}

