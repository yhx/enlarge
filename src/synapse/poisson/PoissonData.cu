#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "PoissonData.h"

void *cudaMallocPoisson()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(PoissonData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(PoissonData)*1));
	return ret;
}

__global__ void curand_setup_poisson_init_state(PoissonData *data)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int idx = 0; idx < data->num; idx += blockDim.x * gridDim.x) {
		int sid = tid + idx;
		if (sid < data->num) {
			curand_init(666, sid, 0, &(data->pState[sid]));
		}
	}
}

void *cudaAllocPoisson(void *pCPU, size_t num)
{
	void *ret = cudaMallocPoisson();
	void *tmp = cudaAllocPoissonPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(PoissonData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	curand_setup_poisson_init_state<<<1, 128>>>((PoissonData *)ret);
	return ret;
}

void *cudaAllocPoissonPara(void *pCPU, size_t num)
{
	PoissonData *p = (PoissonData*)pCPU;
	PoissonData *ret = (PoissonData*)malloc(sizeof(PoissonData)*1);
	memset(ret, 0, sizeof(PoissonData)*1);
	memcpy(ret, p, sizeof(PoissonData)*1);
	
	// copy poisson synapse weight from CPU to GPU 
	checkCudaErrors(cudaMalloc((void**)&(ret->pWeight), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pWeight, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pWeight, p->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));
	
	// copy poisson mean from CPU to GPU 
	checkCudaErrors(cudaMalloc((void**)&(ret->pMean), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pMean, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pMean, p->pMean, sizeof(real)*num, cudaMemcpyHostToDevice));

	// allocate curandState in GPU
	checkCudaErrors(cudaMalloc((void**)&(ret->pState), sizeof(curandState)*num));
	checkCudaErrors(cudaMemset(ret->pState, 0, sizeof(curandState)*num));

	return (void *)ret;
}

int cudaFetchPoisson(void *pCPU, void *pGPU, size_t num)
{
	PoissonData *pTmp = (PoissonData*)malloc(sizeof(PoissonData)*1);
	memset(pTmp, 0, sizeof(PoissonData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(PoissonData)*1, cudaMemcpyDeviceToHost));

	cudaPoissonParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaPoissonParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	PoissonData *pC = (PoissonData*)pCPU;
	PoissonData *pG = (PoissonData*)pGPU;

	//checkCudaErrors(cudaMemcpy(pG->pDst, pC->pDst, sizeof(int)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(pG->pWeight, pC->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pMean, pC->pMean, sizeof(real)*num, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(pG->pState, pC->pState, sizeof(curandState)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaPoissonParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	PoissonData *pC = (PoissonData*)pCPU;
	PoissonData *pG = (PoissonData*)pGPU;

	// checkCudaErrors(cudaMemcpy(pC->pDst, pG->pDst, sizeof(int)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pWeight, pG->pWeight, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pMean, pG->pMean, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pState, pG->pState, sizeof(curandState)*num, cudaMemcpyDeviceToHost));

	return 0;
}

int cudaFreePoisson(void *pGPU)
{
	PoissonData *tmp = (PoissonData*)malloc(sizeof(PoissonData)*1);
	memset(tmp, 0, sizeof(PoissonData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(PoissonData)*1, cudaMemcpyDeviceToHost));
	cudaFreePoissonPara(tmp);
	free(tmp);
	tmp = NULL;
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreePoissonPara(void *pGPU)
{
	PoissonData *p = (PoissonData*)pGPU;
	// cudaFree(p->pDst);
	// p->pDst = NULL;

	cudaFree(p->pWeight);
	cudaFree(p->pMean);
	cudaFree(p->pState);
	p->pWeight = NULL;
	p->pMean = NULL;
	p->pState = NULL;

	return 0;
}

