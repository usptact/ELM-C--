#include <cmath>
#include <iostream>

using namespace std;

// compare function
int compare( const void *a, const void *b ) {

	double eps = 0.00000001;
	double diff = *(double *) a - *(double *) b;

	if ( abs(diff) <= eps )
		return 0;

	if ( diff < 0 )
		return -1;
	else
		return 1;
}

int main() {

	double d[10] = { 0.1, 5.3, 4.2, 1.1, 0.9, 6.5, 0.1, 7.0, 5.9, 2.4 };

	qsort( d, 10, sizeof(double), compare );

	for ( int i = 0 ; i < 10 ; i++ )
		cout << d[i] << " ";

	cout << endl;

	return 0;
}
