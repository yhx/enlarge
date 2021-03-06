
#include <stdlib.h>
#include <assert.h>

#include "../gpu_utils/mem_op.h"
#include "Connection.h"

Connection * cudaAllocConnection(Connection * pCPU)
{
	size_t nNum = pCPU->nNum;
	size_t sNum = pCPU->sNum;
	size_t length = (pCPU->maxDelay - pCPU->minDelay + 1) * nNum;

	Connection * pGPU = NULL;
	Connection *pTmp = (Connection*)malloc(sizeof(Connection));
	pTmp->nNum = nNum;
	pTmp->sNum = sNum;
	pTmp->maxDelay = pCPU->maxDelay;
	pTmp->minDelay = pCPU->minDelay;

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayStart), sizeof(size_t)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayStart, pCPU->pDelayStart, sizeof(size_t)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNum), sizeof(size_t)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayNum, pCPU->pDelayNum, sizeof(size_t)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pSidMap), sizeof(size_t)*sNum));
	checkCudaErrors(cudaMemcpy(pTmp->pSidMap, pCPU->pSidMap, sizeof(size_t)*sNum, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(pTmp->dst), sizeof(size_t)*sNum));
	checkCudaErrors(cudaMemcpy(pTmp->dst, pCPU->dst, sizeof(size_t)*sNum, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayStartRev), sizeof(size_t)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayStartRev, pCPU->pDelayStartRev, sizeof(size_t)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNumRev), sizeof(size_t)*length));
	checkCudaErrors(cudaMemcpy(pTmp->pDelayNumRev, pCPU->pDelayNumRev, sizeof(size_t)*length, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pTmp->pSidMapRev), sizeof(size_t)*sNum));
	checkCudaErrors(cudaMemcpy(pTmp->pSidMapRev, pCPU->pSidMapRev, sizeof(size_t)*sNum, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(pGPU), sizeof(Connection)));
	checkCudaErrors(cudaMemcpy(pGPU, pTmp, sizeof(Connection), cudaMemcpyHostToDevice));

	free(pTmp);
	pTmp = NULL;

	return pGPU;
}

int cudaFetchConnection(Connection *pCPU, Connection *pGPU)
{
	size_t nNum = pCPU->nNum;
	size_t sNum = pCPU->sNum;
	size_t length = (pCPU->maxDelay - pCPU->minDelay + 1) * nNum;

	Connection *pTmp = (Connection*)malloc(sizeof(Connection));
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(Connection), cudaMemcpyDeviceToHost));
	assert(nNum == pTmp->nNum);
	assert(sNum == pTmp->sNum);
	assert(pCPU->maxDelay == pTmp->maxDelay);
	assert(pCPU->minDelay == pTmp->minDelay);

	checkCudaErrors(cudaMemcpy(pCPU->pDelayStart, pTmp->pDelayStart, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pCPU->pDelayNum, pTmp->pDelayNum, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pCPU->pSidMap, pTmp->pSidMap, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pCPU->dst, pTmp->dst, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));


	checkCudaErrors(cudaMemcpy(pCPU->pDelayStartRev, pTmp->pDelayStartRev, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pCPU->pDelayNumRev, pTmp->pDelayNumRev, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pCPU->pSidMapRev, pTmp->pSidMapRev, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));

	free(pTmp);
	pTmp = NULL;

	return 0;
}

int cudaFreeConnection(Connection *pGPU)
{
	Connection * pTmp = (Connection*)malloc(sizeof(Connection));
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(Connection), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(pTmp->pDelayStart));
	checkCudaErrors(cudaFree(pTmp->pDelayNum));
	checkCudaErrors(cudaFree(pTmp->pSidMap));
	checkCudaErrors(cudaFree(pTmp->dst));
	checkCudaErrors(cudaFree(pTmp->pDelayStartRev));
	checkCudaErrors(cudaFree(pTmp->pDelayNumRev));
	checkCudaErrors(cudaFree(pTmp->pSidMapRev));
	free(pTmp);
	pTmp = NULL;
	return 0;
}
