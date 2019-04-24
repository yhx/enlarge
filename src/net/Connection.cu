
#include <stdlib.h>

#include "../gpu_utils/mem_op.h"
#include "Connection.h"

Connection * cudaAllocConnection(Connection * pCPU)
{
	int nNum = pCPU->nNum;
	int sNum = pCPU->sNum;
	int length = (pCPU->maxDelay - pCPU->minDelay + 1) * nNum;

	Connection * pGPU = NULL;
	Connection *pTmp = (Connection*)malloc(sizeof(Connection));
	pTmp->nNum = nNum;
	pTmp->nNum = sNum;
	pTmp->maxDelay = pCPU->maxDelay;
	pTmp->minDelay = pCPU->minDelay;

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayStart), sizeof(int)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayStart, pCPU->pDelayStart, sizeof(int)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNum), sizeof(int)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayNum, pCPU->pDelayNum, sizeof(int)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayStartRev), sizeof(int)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayStartRev, pCPU->pDelayStartRev, sizeof(int)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNumRev), sizeof(int)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayNumRev, pCPU->pDelayNumRev, sizeof(int)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNumRev), sizeof(int)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayNumRev, pCPU->pDelayNumRev, sizeof(int)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pSidMapRev), sizeof(int)*sNum));
	checkCudaErrors(cudaMemcpy(pTmp->pSidMapRev, pCPU->pSidMapRev, sizeof(int)*sNum, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pGPU), sizeof(Connection)));
	checkCudaErrors(cudaMemcpy(pGPU, pTmp, sizeof(Connection), cudaMemcpyHostToDevice));
	free(pTmp);
	pTmp = NULL;

	return pGPU;
}
