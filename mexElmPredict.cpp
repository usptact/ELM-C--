// Predicts class scores using a trained ELM model.
//
//	scores = mexElmPredict( inW, bias, outW, X );
//
// INPUT :
//	inW 	- input weights matrix (trained model)
//	bias 	- bias vector (trained model)
//	outW 	- output weights matrix (trained model)
//	X 	- samples matrix (vectors in columns)
//
// OUTPUT :
//	scores 	- prediction scores for samples in matrix X
//

#include "elml.hpp"

#include "mex.h"
#include "matrix.h"

// input variable order numbers
#define INW 0
#define BIAS 1
#define OUTW 2
#define XDATA 3

// output variable order numbers
#define SCRS 0

using namespace std;

// entry point
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

	if ( nrhs != 4 )
		mexErrMsgTxt( "Four input arguments required." );

	//
	// CHECK INPUT TYPES
	//

	if ( !(mxGetClassID(prhs[INW]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Matrix inW must be DOUBLE type." );

	if ( !(mxGetClassID(prhs[BIAS]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Matrix bias must be DOUBLE type." );

	if ( !(mxGetClassID(prhs[OUTW]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Matrix outW must be DOUBLE type." );

	if ( !(mxGetClassID(prhs[XDATA]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Matrix X must be DOUBLE type." );

	//
	// GET POINTERS TO INPUT DATA
	//

	// get pointers to trained model matrices
	double *inW_ptr = (double *) mxGetPr( prhs[INW] );
	double *bias_ptr = (double *) mxGetPr( prhs[BIAS] );
	double *outW_ptr = (double *) mxGetPr( prhs[OUTW] );

	// get pointers to samples matrix
	double *X_ptr = (double *) mxGetPr( prhs[XDATA] );

	//
	// GET INFO ABOUT INPUT DATA
	//

	int Xdims = static_cast<int>( mxGetM(prhs[XDATA]) );
	int Xnsmp = static_cast<int>( mxGetN(prhs[XDATA]) );

	int inW_rows = static_cast<int>( mxGetM(prhs[INW]) );
	int inW_cols = static_cast<int>( mxGetN(prhs[INW]) );

	int bias_rows = static_cast<int>( mxGetM(prhs[BIAS]) );

	int outW_rows = static_cast<int>( mxGetM(prhs[OUTW]) );
	int outW_cols = static_cast<int>( mxGetN(prhs[OUTW]) );

	//
	// BUILD EIGEN MATRIX OBJECTS
	//

	MatrixXd inW = Map<MatrixXd>( inW_ptr, inW_rows, inW_cols );
	MatrixXd bias = Map<MatrixXd>( bias_ptr, bias_rows, 1 );
	MatrixXd outW = Map<MatrixXd>( outW_ptr, outW_rows, outW_cols );

	//
	// PREDICTION
	//

	MatrixXd mScores;

	// launch prediction
	int code = elmPredict( X_ptr, Xdims, Xnsmp,
		    mScores,
		    inW, bias, outW );

	if ( code != 0 )
		mexErrMsgTxt( "Failed to predict class scores." );

	//
	// OUTPUT PREDICTION SCORE TO MATLAB
	//

	// allocate output matrix
	mxArray *scores_matlab = mxCreateDoubleMatrix( mScores.rows(), mScores.cols(), mxREAL );

	// get C pointer to output matrix
	double *scrs_c = mxGetPr( scores_matlab );

	// get the scores out from the Eigen matrix
	Map<MatrixXd>( scrs_c, mScores.rows(), mScores.cols() ) =  mScores;

	// assign output
	plhs[SCRS] = scores_matlab;

}

