
#include <stdlib.h>
#include <assert.h>

#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "Connection.h"

Connection * cudaAllocConnection(Connection * pCPU)
{
	size_t nNum = pCPU->nNum;
	size_t sNum = pCPU->sNum;
	size_t length = (pCPU->maxDelay - pCPU->minDelay + 1) * nNum;

	Connection * pGPU = NULL;
	Connection *pTmp = malloc_c<Connection>();
	pTmp->nNum = nNum;
	pTmp->sNum = sNum;
	pTmp->maxDelay = pCPU->maxDelay;
	pTmp->minDelay = pCPU->minDelay;

	pTmp->pDelayStart = TOGPU(pCPU->pDelayStart, length+1);
	pTmp->pDelayNum = TOGPU(pCPU->pDelayNum, length);
	pTmp->pSidMap = TOGPU(pCPU->pSidMap, sNum);
	pTmp->dst = TOGPU(pCPU->dst, sNum);

	// pTmp->pDelayStartRev = copyToGPU(pCPU->pDelayStartRev, length+1);

	// checkCudaErrors(cudaMalloc((void**)&(pTmp->pDelayNumRev), sizeof(size_t)*length));
	// checkCudaErrors(cudaMemcpy(pTmp->pDelayNumRev, pCPU->pDelayNumRev, sizeof(size_t)*length, cudaMemcpyHostToDevice));

	// checkCudaErrors(cudaMalloc((void**)&(pTmp->pSidMapRev), sizeof(size_t)*sNum));
	// checkCudaErrors(cudaMemcpy(pTmp->pSidMapRev, pCPU->pSidMapRev, sizeof(size_t)*sNum, cudaMemcpyHostToDevice));

	pGPU = TOGPU(pTmp, 1);

	free(pTmp);
	pTmp = NULL;

	return pGPU;
}

int cudaFetchConnection(Connection *pCPU, Connection *pGPU)
{
	size_t nNum = pCPU->nNum;
	size_t sNum = pCPU->sNum;
	size_t length = (pCPU->maxDelay - pCPU->minDelay + 1) * nNum;

	Connection *pTmp = malloc_c<Connection>();
	COPYFROMGPU(pTmp, pGPU, 1);
	// checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(Connection), cudaMemcpyDeviceToHost));
	assert(nNum == pTmp->nNum);
	assert(sNum == pTmp->sNum);
	assert(pCPU->maxDelay == pTmp->maxDelay);
	assert(pCPU->minDelay == pTmp->minDelay);

	COPYFROMGPU(pCPU->pDelayStart, pTmp->pDelayStart, length+1);
	COPYFROMGPU(pCPU->pDelayNum, pTmp->pDelayNum, length);
	COPYFROMGPU(pCPU->pSidMap, pTmp->pSidMap, sNum);
	COPYFROMGPU(pCPU->dst, pTmp->dst, sNum);
	// checkCudaErrors(cudaMemcpy(pCPU->pDelayStart, pTmp->pDelayStart, sizeof(size_t)*(length+1), cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pCPU->pDelayNum, pTmp->pDelayNum, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pCPU->pSidMap, pTmp->pSidMap, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pCPU->dst, pTmp->dst, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));


	// checkCudaErrors(cudaMemcpy(pCPU->pDelayStartRev, pTmp->pDelayStartRev, sizeof(size_t)*(length+1), cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pCPU->pDelayNumRev, pTmp->pDelayNumRev, sizeof(size_t)*length, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pCPU->pSidMapRev, pTmp->pSidMapRev, sizeof(size_t)*sNum, cudaMemcpyDeviceToHost));

	free(pTmp);
	pTmp = NULL;

	return 0;
}

int cudaFreeConnection(Connection *pGPU)
{
	Connection * pTmp = malloc_c<Connection>();

	COPYFROMGPU(pTmp, pGPU, 1);
	// checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(Connection), cudaMemcpyDeviceToHost));
	gpuFree(pTmp->pDelayStart);
	gpuFree(pTmp->pDelayNum);
	gpuFree(pTmp->pSidMap);
	gpuFree(pTmp->dst);
	// checkCudaErrors(cudaFree(pTmp->pDelayStartRev));
	// checkCudaErrors(cudaFree(pTmp->pDelayNumRev));
	// checkCudaErrors(cudaFree(pTmp->pSidMapRev));
	free(pTmp);
	pTmp = NULL;
	return 0;
}
