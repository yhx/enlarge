
#include "../../msg_utils/helper/helper_gpu.h"
#include "gpu_utils.h"

void print_gmem(const char *msg)
{
	size_t fmem = 0, tmem = 0;
	checkCudaErrors(cudaMemGetInfo(&fmem, &tmem));
	printf("%s GMEM used: %lfGB\n", msg, static_cast<double>((tmem - fmem)/1024.0/1024.0/1024.0));
}
