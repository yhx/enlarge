
#include "../../include/BSim.h"
#include "info.h"

using namespace std;

int main(int argc, char **argv)
{
	real * weight = getRandomArray((real)40e-3, N*N);
	real * weight2 = getRandomArray((real)90e-3, N*N);
	real * weight3 = getRandomArray((real)150e-3, N*N);
	real * weight4 = getRandomArray((real)300e-3, N*N);
	real * delay = getConstArray((real)1e-3, N*N);

	saveArray(W1_NAME, weight, N*N);
	saveArray(W2_NAME, weight2, N*N);
	saveArray(W3_NAME, weight3, N*N);
	saveArray(W4_NAME, weight4, N*N);
	saveArray(D_NAME, delay, N*N);

	return 0;
}
