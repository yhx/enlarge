
#include "../../include/BSim.h"
#include "info.h"

using namespace std;

int main(int argc, char **argv)
{
	real * weight = getRandomArray((real)20e-3, N*N);
	real * weight2 = getRandomArray((real)15e-3, N*N);
	real * delay = getConstArray((real)1e-3, N*N);

	saveArray(W1_NAME, weight, N*N);
	saveArray(W2_NAME, weight2, N*N);
	saveArray(D_NAME, delay, N*N);

	return 0;
}
