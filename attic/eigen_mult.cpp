//
// Example generates some random data in the Eigen matrix and outputs it.
//

#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

int main() {

	const int nCols = 5;
	const int nRows = 4;

	MatrixXd X( nRows, nCols );
	X.setRandom();

	MatrixXd Y( nCols, nRows );
	Y.setRandom();

	MatrixXd Z = X.matrix() * Y.matrix();

	cout << Z << endl;

	return 0;
}

