//
// This example demonstrates the usage of Map in Eigen framework.
// User-provided data is used to populate a Eigen matrix.
//

#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;

int main() {

	// user data
	double data[6] = { 1, 2, 3, 4, 5 , 6 };

	// creating a 3x2 matrix from user data
	MatrixXd mat = Map<MatrixXd>( data, 3, 2 );

	// output it to see what is inside
	cout << mat << endl;

	return 0;
}

