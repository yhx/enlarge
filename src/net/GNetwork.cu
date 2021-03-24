#include <assert.h>

#include "../neuron/lif/LIFData.h"
#include "../base/TypeFunc.h"
#include "../utils/macros.h"
#include "../utils/helper_c.h"
#include "../gpu_utils/helper_gpu.h"
#include "GNetwork.h"

GNetwork* copyGNetworkToGPU(GNetwork *pCpuNet)
{
	if (pCpuNet == NULL) {
		printf("NULL POINTER: GNETWORK\n");
		return NULL;
	}

	size_t nTypeNum = pCpuNet->nTypeNum;
	size_t sTypeNum = pCpuNet->sTypeNum;

	GNetwork *tmp = allocGNetwork(nTypeNum, sTypeNum);

	// tmp->maxDelay = pCpuNet->maxDelay;
	// tmp->minDelay = pCpuNet->minDelay;

	for (size_t i=0; i<nTypeNum; i++) {
		tmp->pNTypes[i] = pCpuNet->pNTypes[i];
		tmp->pNeuronNums[i] = pCpuNet->pNeuronNums[i];
		tmp->bufferOffsets[i] = pCpuNet->bufferOffsets[i];
	}
	tmp->pNeuronNums[nTypeNum] = pCpuNet->pNeuronNums[nTypeNum];
	tmp->bufferOffsets[nTypeNum] = pCpuNet->bufferOffsets[nTypeNum];

	for (size_t i=0; i<sTypeNum; i++) {
		tmp->pSTypes[i] = pCpuNet->pSTypes[i];
		tmp->pSynapseNums[i] = pCpuNet->pSynapseNums[i];
	}
	tmp->pSynapseNums[sTypeNum] = pCpuNet->pSynapseNums[sTypeNum];


	for (size_t i=0; i<nTypeNum; i++) {
		tmp->ppNeurons[i] = cudaAllocType[pCpuNet->pNTypes[i]](pCpuNet->ppNeurons[i], pCpuNet->pNeuronNums[i+1]-pCpuNet->pNeuronNums[i]);
	}

	for (size_t i=0; i<sTypeNum; i++) {
		tmp->ppSynapses[i] = cudaAllocType[pCpuNet->pSTypes[i]](pCpuNet->ppSynapses[i], pCpuNet->pSynapseNums[i+1]-pCpuNet->pSynapseNums[i]);
	}

	tmp->ppConnections = (Connection **)malloc(sizeof(Connection*)*sTypeNum);
	for (size_t i=0; i<sTypeNum; i++) {
		tmp->ppConnections[i] = cudaAllocConnection(pCpuNet->ppConnections[i]);
	}

	return tmp;
}

int fetchGNetworkFromGPU(GNetwork *pCpuNet, GNetwork *pGpuNet)
{
	if (pCpuNet == NULL || pGpuNet == NULL) {
		printf("NULL POINTER: FETCH GNETWORK\n");
		return -1;
	}

	size_t nTypeNum = pGpuNet->nTypeNum;
	size_t sTypeNum = pGpuNet->sTypeNum;

	assert(pCpuNet->nTypeNum == nTypeNum);
	assert(pCpuNet->sTypeNum == sTypeNum);

	for (size_t i=0; i<nTypeNum; i++) {
		pCpuNet->pNTypes[i] = pGpuNet->pNTypes[i];
		pCpuNet->pNeuronNums[i] = pGpuNet->pNeuronNums[i];
		pCpuNet->bufferOffsets[i] = pGpuNet->bufferOffsets[i];
	}
	pCpuNet->pNeuronNums[nTypeNum] = pGpuNet->pNeuronNums[nTypeNum];
	pCpuNet->bufferOffsets[nTypeNum] = pGpuNet->bufferOffsets[nTypeNum];

	for (size_t i=0; i<sTypeNum; i++) {
		pCpuNet->pSTypes[i] = pGpuNet->pSTypes[i];
		pCpuNet->pSynapseNums[i] = pGpuNet->pSynapseNums[i];
	}
	pCpuNet->pSynapseNums[sTypeNum] = pGpuNet->pSynapseNums[sTypeNum];

	//TODO support multitype N and S
	for (size_t i=0; i<nTypeNum; i++) {
		//TODO: cudaFetchType
		cudaFetchType[pCpuNet->pNTypes[i]](pCpuNet->ppNeurons[i], pGpuNet->ppNeurons[i], pCpuNet->pNeuronNums[i+1]-pCpuNet->pNeuronNums[i]);
	}
	for (size_t i=0; i<sTypeNum; i++) {
		//TODO: cudaFetchType
		cudaFetchType[pCpuNet->pSTypes[i]](pCpuNet->ppSynapses[i], pGpuNet->ppSynapses[i], pCpuNet->pSynapseNums[i+1]-pCpuNet->pSynapseNums[i]);
	}

	for (size_t i=0; i<sTypeNum; i++) {
		cudaFetchConnection(pCpuNet->ppConnections[i], pGpuNet->ppConnections[i]);
	}
	return 0;
}

int freeGNetworkGPU(GNetwork *pGpuNet)
{
	GNetwork *pTmp = pGpuNet;

	size_t nTypeNum = pTmp->nTypeNum;
	size_t sTypeNum = pTmp->sTypeNum;

	free_c(pGpuNet->pNTypes);
	free_c(pGpuNet->pSTypes);

	free_c(pGpuNet->pNeuronNums);
	free_c(pGpuNet->pSynapseNums);

	free_c(pGpuNet->bufferOffsets);

	for (size_t i=0; i<nTypeNum; i++) {
		cudaFreeType[pTmp->pNTypes[i]](pTmp->ppNeurons[i]);
	}

	for (size_t i=0; i<sTypeNum; i++) {
		cudaFreeType[pTmp->pSTypes[i]](pTmp->ppSynapses[i]);
	}

	for (size_t i=0; i<sTypeNum; i++) {
		cudaFreeConnection(pTmp->ppConnections[i]);
	}

	free_c(pTmp->ppNeurons);
	free_c(pTmp->ppSynapses);
	free_c(pTmp->ppConnections);
	free_c(pTmp);

	return 0;
}



// int checkGNetworkGPU(GNetwork *g, GNetwork *c)
// {
// 	// TODO finish check
// 	int ret = -1;
// 
// 	CHECK(g, c, nTypeNum);
// 	CHECK(g, c, sTypeNum);
// 	CHECK(g, c, pNeuronNums);
// 	CHECK(g, c, pSynapseNums);
// 
// 	CHECK(g, c, pNTypes);
// 	CHECK(g, c, pSTypes);
// 
// 	ret = 1;
// 
// 	//size_t totalNeuronNum = g->pNeuronNums[g->nTypeNum+1];
// 	//size_t totalSynapseNum = g->pSynapseNums[g->sTypeNum+1];
// 	// Connection p;
// 	// checkCudaErrors(cudaMemcpy(&p, g->pConnection, sizeof(Connection), cudaMemcpyDeviceToHost));
// 
// 	// CHECK_GPU_TO_CPU_ARRAY(p.delayStart, c->pConnection->delayStart, sizeof(size_t)*(c->pConnection->nNum)*(maxDelay-minDelay+1));
// 	// CHECK_GPU_TO_CPU_ARRAY(p.delayNum, c->pConnection->delayNum, sizeof(size_t)*(c->pConnection->nNum)*(maxDelay-minDelay+1));
// 
// 	ret = 2;
// 
// 	return ret;
// }
