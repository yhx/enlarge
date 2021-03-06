#include <stdlib.h>
#include <string.h>
#include "../../third_party/cuda/helper_cuda.h"
#include "GStaticSynapse.h"

void *cudaMallocStatic()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(GStaticSynapse)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(GStaticSynapse)*1));
	return ret;
}

void *cudaAllocStatic(void *pCPU, int num)
{
	void *ret = cudaMallocStatic();
	void *tmp = cudaAllocStaticPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(GStaticSynapse)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocStaticPara(void *pCPU, int num)
{
	GStaticSynapse *p = (GStaticSynapse*)pCPU;
	GStaticSynapse *ret = (GStaticSynapse*)malloc(sizeof(GStaticSynapse)*1);
	memset(ret, 0, sizeof(GStaticSynapse)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->pWeight), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pWeight, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pWeight, p->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pDst), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pDst, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pDst, p->pDst, sizeof(int)*num, cudaMemcpyHostToDevice));

	return ret;
}

int cudaFetchStatic(void *pCPU, void *pGPU, int num)
{
	GStaticSynapse *pTmp = (GStaticSynapse*)malloc(sizeof(GStaticSynapse)*1);
	memset(pTmp, 0, sizeof(GStaticSynapse)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(GStaticSynapse)*1, cudaMemcpyDeviceToHost));

	cudaStaticParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaStaticParaToGPU(void *pCPU, void *pGPU, int num)
{
	GStaticSynapse *pC = (GStaticSynapse*)pCPU;
	GStaticSynapse *pG = (GStaticSynapse*)pGPU;

	checkCudaErrors(cudaMemcpy(pG->pWeight, pC->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(pG->pDst, pC->pDst, sizeof(int)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaStaticParaFromGPU(void *pCPU, void *pGPU, int num)
{
	GStaticSynapse *pC = (GStaticSynapse*)pCPU;
	GStaticSynapse *pG = (GStaticSynapse*)pGPU;

	checkCudaErrors(cudaMemcpy(pC->pWeight, pG->pWeight, sizeof(real)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pDst, pG->pDst, sizeof(int)*num, cudaMemcpyDeviceToHost));

	return 0;
}

int cudaFreeStatic(void *pGPU)
{
	GStaticSynapse *tmp = (GStaticSynapse*)malloc(sizeof(GStaticSynapse)*1);
	memset(tmp, 0, sizeof(GStaticSynapse)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(GStaticSynapse)*1, cudaMemcpyDeviceToHost));
	cudaFreeStaticPara(tmp);
	free(tmp);
	tmp = NULL;
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeStaticPara(void *pGPU)
{
	GStaticSynapse *p = (GStaticSynapse*)pGPU;
	cudaFree(p->pWeight);
	p->pWeight = NULL;

	cudaFree(p->pDst);
	p->pDst = NULL;

	return 0;
}

