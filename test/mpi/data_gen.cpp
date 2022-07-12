
#include "../../include/BSim.h"
#include "info.h"

using namespace std;

int main(int argc, char **argv)
{
	real *weight0 = (real*)malloc_c<real>(N * N);
	real *weight1 = (real*)malloc_c<real>(N * N);
	real *weight2 = (real*)malloc_c<real>(N * N);

	for (int i=0; i<N*N; i++) {
		weight0[i] = 0.002 + (double)(i)/(double)(N*N) * 0.02;
		weight1[i] = 0.022 - (double)(i)/(double)(N*N) * 0.02;
		weight2[i] = -0.022 + (double)(i)/(double)(N*N) * 0.02;
	}
	
	real * delay1 = getConstArray((real)1e-4, N*N);
	real * delay2 = getConstArray((real)2e-4, N*N);

	saveArray(W0_NAME, weight0, N*N);
	saveArray(W1_NAME, weight1, N*N);
	saveArray(W2_NAME, weight2, N*N);
	saveArray(D1_NAME, delay1, N*N);
	saveArray(D2_NAME, delay2, N*N);

	return 0;
}
