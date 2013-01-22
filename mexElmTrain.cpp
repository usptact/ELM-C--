// Trains an ELM model on the provided dataset.
//
//	[inW bias outW] = mexElmTrain( X, Y [, nhn, C ] );
//
// INPUT :
//	X 	- samples matrix (samples in columns)
//	Y 	- labels vector
//	nhn 	- number of hidden neurons (default: dims / 2)
//	C 	- regularization parameter (default: 1)
//
// OUTPUT :
//	inW 	- output weights matrix
//	bias 	- bias vector
//	outW 	- output weights matrix
//

#include "elml.hpp"

#include "mex.h"
#include "matrix.h"

// input variable order numbers
#define XDATA 0
#define YDATA 1
#define NHIDD 2
#define CPARA 3

// output variable order numbers
#define INW 0
#define BIAS 1
#define OUTW 2

using namespace std;

// entry point
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

	if ( nrhs < 2 )
		mexErrMsgTxt( "At least two input arguments required." );
	
	//
	// CHECK INPUT TYPES
	//

	// data must be in double
	if ( !(mxGetClassID(prhs[XDATA]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Data matrix must be DOUBLE type." );

	// labels vector must be integer
	if ( !(mxGetClassID(prhs[YDATA]) == mxDOUBLE_CLASS) )
		mexErrMsgTxt( "Labels vector must be DOUBLE type." );

	//
	// GET POINTERS TO INPUT DATA
	//

	// get pointer to samples
	double *X_ptr = (double *) mxGetPr( prhs[XDATA] );

	// get pointer to labels
	double *Y_ptr = (double *) mxGetPr( prhs[YDATA] );

	// get pointer to number of hidden neurons
	if ( nrhs > 2 )
		int *nhn_ptr = (int *) mxGetData( prhs[NHIDD] );

	// get pointer to regularization parameter
	if ( nrhs > 3 )
		double *C_ptr = (double *) mxGetPr( prhs[CPARA] );

	//
	// GET INFO ABOUT INPUT DATA
	//

	// dimensionality and number of samples (X must be a matrix)
	int ndimsX = static_cast<int>( mxGetM(prhs[XDATA]) );
	int nsmpX = static_cast<int>( mxGetN(prhs[XDATA]) );

	// dimensionality and number of labels (Y must be a vector!)
	int ndimsY = static_cast<int>( mxGetM(prhs[YDATA]) );
	int nsmpY = static_cast<int>( mxGetN(prhs[YDATA]) );

	// check that Y is a column vector
	if ( nsmpY > 1 )
		mexErrMsgTxt( "Labels vector must be a column vector." );

	// check that number of samples and the length of Y vector match
	if ( nsmpX != ndimsY )
		mexErrMsgTxt( "Data matrix column and labels vector length mismatch." );

	// read parameter values
	double *tmp = mxGetPr( prhs[CPARA] );
	const double C = static_cast<double>( *tmp );	
	tmp = mxGetPr( prhs[NHIDD] );
	const int nhn = static_cast<int>( *tmp );

	//
	// MODEL TRAINING
	//

	MatrixXd inW;
	MatrixXd bias;
	MatrixXd outW;

	// launch training procedure
	int code = elmTrain( X_ptr, ndimsX, nsmpX,
			     Y_ptr,
			     nhn, C,
			     inW, bias, outW );

	if ( code != 0 )
		mexErrMsgTxt( "Failed to train a model." );

	//
	// OUTPUT TRAINED MODEL TO MATLAB
	//

	// allocate output arrays
	mxArray *inW_matlab = mxCreateDoubleMatrix( inW.rows(), inW.cols(), mxREAL );
	mxArray *bias_matlab = mxCreateDoubleMatrix( bias.rows(), bias.cols(), mxREAL );
	mxArray *outW_matlab = mxCreateDoubleMatrix( outW.rows(), outW.cols(), mxREAL );

	// get C pointers to the output arrays
	double *inW_c = mxGetPr( inW_matlab );
	double *bias_c = mxGetPr( bias_matlab );
	double *outW_c = mxGetPr( outW_matlab );
	
	// get the data from Eigen matrices inside Matlab's
	Map<MatrixXd>( inW_c, inW.rows(), inW.cols() ) = inW;
	Map<MatrixXd>( bias_c, bias.rows(), bias.cols() ) = bias;
	Map<MatrixXd>( outW_c, outW.rows(), outW.cols() ) = outW;

	// assign outputs
	plhs[INW] = inW_matlab;
	plhs[BIAS] = bias_matlab;
	plhs[OUTW] = outW_matlab;

}

