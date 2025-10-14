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
//
// -----------------------------------------------
// Provides Extreme Learning Machine algorithm.
// -----------------------------------------------
//
// Refer to:
// G.-B. Huang et al., "Extreme Learning Machine for Regression and Multiclass Classification," IEEE TSMC, vol. 42, no. 2, pp. 513-529, 2012.
//
// Original implementation: Vladislavs D.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <map>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace Eigen;

#ifndef ELM_H
#define ELM_H

int compare( const void *a, const void *b );

// Build target matrix from labels (supports arbitrary label values)
MatrixXd buildTargetMatrix( double *Y, int nLabels );

// entry function to train the ELM model
// Uses sigmoid activation function: f(x) = 1 / (1 + exp(-x))
// INPUT: X (dims x nsmp), Y (nsmp labels), nhn (hidden neurons), C (regularization)
// OUTPUT: inW (nhn x dims), bias (nhn x 1), outW (nhn x nclasses)
// RETURNS: 0 on success, -1 on error
template <typename Derived>
int elmTrain( double *X, int dims, int nsmp, 
       	      double *Y,
	      const int nhn, const double C,
	      MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW ) {

	// Input validation
	if ( X == nullptr || Y == nullptr ) {
		cerr << "Error: NULL pointer passed to elmTrain" << endl;
		return -1;
	}
	if ( dims <= 0 || nsmp <= 0 || nhn <= 0 || C <= 0 ) {
		cerr << "Error: Invalid parameters (dims=" << dims << ", nsmp=" << nsmp 
		     << ", nhn=" << nhn << ", C=" << C << ")" << endl;
		return -1;
	}

	try {
		// map the samples into the matrix object
		MatrixXd mX = Map<MatrixXd>( X, dims, nsmp );

		// build target matrix
		MatrixXd mTargets = buildTargetMatrix( Y, nsmp );

		// generate random input weight matrix - inW
		inW = MatrixXd::Random( nhn, dims );

		// generate random bias vectors
		bias = MatrixXd::Random( nhn, 1 );

		// compute the pre-H matrix
		MatrixXd preH = inW * mX + bias.replicate( 1, nsmp );

		// compute hidden neuron output (sigmoid activation)
		MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

		// build matrices to solve Ax = b
		MatrixXd A = (MatrixXd::Identity(nhn,nhn)).array()*(1/C) + (H * H.transpose()).array();
		MatrixXd b = H * mTargets.transpose();
		
		// solve the output weights using Cholesky decomposition
		LLT<MatrixXd> lltOfA(A);
		if ( lltOfA.info() != Success ) {
			cerr << "Error: Matrix decomposition failed" << endl;
			return -1;
		}
		outW = lltOfA.solve( b );
		
		return 0;
	}
	catch ( const exception& e ) {
		cerr << "Error in elmTrain: " << e.what() << endl;
		return -1;
	}

}

// entry function to predict class labels using the trained ELM model on test data
// INPUT : X (dims x nsmp), inW (nhn x dims), bias (nhn x 1), outW (nhn x nclasses)
// OUTPUT : mScores (nclasses x nsmp) - prediction scores for each class
// RETURNS: 0 on success, -1 on error
template <typename Derived>
int elmPredict( double *X, int dims, int nsmp,
		MatrixBase<Derived> &mScores,
		MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW ) {

	// Input validation
	if ( X == nullptr ) {
		cerr << "Error: NULL pointer passed to elmPredict" << endl;
		return -1;
	}
	if ( dims <= 0 || nsmp <= 0 ) {
		cerr << "Error: Invalid parameters (dims=" << dims << ", nsmp=" << nsmp << ")" << endl;
		return -1;
	}
	if ( inW.cols() != dims || bias.rows() != inW.rows() || outW.rows() != inW.rows() ) {
		cerr << "Error: Dimension mismatch in model parameters" << endl;
		return -1;
	}

	try {
		// map the sample into the Eigen's matrix object
		MatrixXd mX = Map<MatrixXd>( X, dims, nsmp );

		// build the pre-H matrix
		MatrixXd preH = inW * mX + bias.replicate( 1, nsmp );

		// apply the sigmoid activation function
		MatrixXd H = ( 1 + (-preH.array()).exp() ).cwiseInverse();

		// compute output scores
		mScores = (H.transpose() * outW).transpose();

		return 0;
	}
	catch ( const exception& e ) {
		cerr << "Error in elmPredict: " << e.what() << endl;
		return -1;
	}
}


// --------------------------
// Helper functions
// --------------------------

// compares two integer values
//int compare( const void* a, const void *b ) {
//	return ( *(int *) a - *(int *) b );
//}

int compare( const void *a, const void *b )
{
	const double *da = (const double *) a;
	const double *db = (const double *) b;
	return (*da > *db) - (*da < *db);
}

// builds 1-of-K target matrix from labels array
// Supports arbitrary label values (not just 1-based sequential integers)
// Normalizes target values to [-1, 1] for better numerical stability
MatrixXd buildTargetMatrix( double *Y, int nLabels ) {

	if ( Y == nullptr || nLabels <= 0 ) {
		throw runtime_error( "Invalid arguments to buildTargetMatrix" );
	}

	// Find unique labels and create mapping
	vector<double> labels( Y, Y + nLabels );
	vector<double> uniqueLabels = labels;
	sort( uniqueLabels.begin(), uniqueLabels.end() );
	uniqueLabels.erase( unique( uniqueLabels.begin(), uniqueLabels.end() ), uniqueLabels.end() );

	int nunique = static_cast<int>( uniqueLabels.size() );
	
	if ( nunique == 0 ) {
		throw runtime_error( "No unique labels found" );
	}

	// Create label to index mapping
	map<double, int> labelToIdx;
	for ( int i = 0; i < nunique; i++ ) {
		labelToIdx[uniqueLabels[i]] = i;
	}

	// Build target matrix (nunique x nLabels)
	MatrixXd targets( nunique, nLabels );
	targets.fill( -1 );  // Fill with -1 by default

	// Set corresponding class to +1
	for ( int i = 0; i < nLabels; i++ ) {
		int idx = labelToIdx[Y[i]];
		targets( idx, i ) = 1;
	}

	return targets;

}

#endif

