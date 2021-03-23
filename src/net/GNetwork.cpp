/* This program is writen by qp09.
 * usually just for fun.
 * 一 四月 18 2016
 */

#include <mpi.h>
#include <assert.h>
#include <iostream>

#include "../utils/utils.h"
#include "../base/TypeFunc.h"
#include "../utils/helper_c.h"

#include "GNetwork.h"

GNetwork * deepcopyGNetwork(GNetwork * net) {
	GNetwork * ret = allocGNetwork(net->nTypeNum, net->sTypeNum);

	for (size_t i=0; i<net->nTypeNum; i++) {
		ret->pNTypes[i] = net->pNTypes[i];
		ret->pNeuronNums[i] = net->pNeuronNums[i];
		ret->ppNeurons[i] = allocType[net->pNTypes[i]](net->pNeuronNums[i+1]-net->pNeuronNums[i]);
	}
	ret->pNeuronNums[net->nTypeNum] = net->pNeuronNums[net->nTypeNum];

	for (size_t i=0; i<net->sTypeNum; i++) {
		ret->pSTypes[i] = net->pSTypes[i];
		ret->pSynapseNums[i] = net->pSynapseNums[i];
		ret->ppSynapses[i] = allocType[net->pSTypes[i]](net->pSynapseNums[i+1]-net->pSynapseNums[i]);
	}
	ret->pSynapseNums[net->sTypeNum] = net->pSynapseNums[net->sTypeNum];

	ret->ppConnections = malloc_c<Connection*>(net->sTypeNum);
	for (size_t i=0; i<net->sTypeNum; i++) {
		ret->ppConnections[i] = allocConnection(net->ppConnections[i]->nNum, net->ppConnections[i]->sNum, net->ppConnections[i]->maxDelay, net->ppConnections[i]->minDelay);
	}

	return ret;
}

GNetwork * allocGNetwork(size_t nTypeNum, size_t sTypeNum) {
	GNetwork *ret = (GNetwork *)malloc(sizeof(GNetwork));
	assert(ret != NULL);

	ret->nTypeNum = nTypeNum;
	ret->sTypeNum = sTypeNum;

	// ret->maxDelay = 1;
	// ret->minDelay = 1e7;

	ret->pNTypes = (Type *)malloc(sizeof(Type)*nTypeNum);
	assert(ret->pNTypes != NULL);
	ret->pSTypes = (Type *)malloc(sizeof(Type)*sTypeNum);
	assert(ret->pSTypes != NULL);

	ret->pNeuronNums = (size_t*)malloc(sizeof(size_t)*(nTypeNum + 1));
	assert(ret->pNeuronNums != NULL);
	ret->pSynapseNums = (size_t*)malloc(sizeof(size_t)*(sTypeNum + 1));
	assert(ret->pSynapseNums != NULL);

	ret->pNeuronNums[0] = 0;
	ret->pSynapseNums[0] = 0;

	ret->ppNeurons = (void **)malloc(sizeof(void*)*nTypeNum);
	assert(ret->ppNeurons != NULL);
	ret->ppSynapses = (void **)malloc(sizeof(void*)*sTypeNum);
	assert(ret->ppSynapses != NULL);

	ret->ppConnections = malloc_c<Connection*>(sTypeNum);

	return ret;
}


void freeGNetwork(GNetwork * network)
{
	for (size_t i=0; i<network->nTypeNum; i++) {
		freeType[network->pNTypes[i]](network->ppNeurons[i]);
	}
	free(network->ppNeurons);

	for (size_t i=0; i<network->sTypeNum; i++) {
		freeType[network->pSTypes[i]](network->ppSynapses[i]);
	}
	free(network->ppSynapses);

	for (size_t i=0; i<network->sTypeNum; i++) {
		freeConnection(network->ppConnections[i]);
	}
	free(network->ppConnections);

	free(network->pNeuronNums);
	free(network->pSynapseNums);

	free(network->pNTypes);
	free(network->pSTypes);
}

int saveGNetwork(GNetwork *net, FILE *f)
{
	fwrite_c(&(net->nTypeNum), 1, f);
	fwrite_c(&(net->sTypeNum), 1, f);
	// fwrite_c(&(net->maxDelay), 1, f);
	// fwrite_c(&(net->minDelay), 1, f);

	fwrite_c(net->pNTypes, net->nTypeNum, f);
	fwrite_c(net->pSTypes, net->sTypeNum, f);
	fwrite_c(net->pNeuronNums, net->nTypeNum+1, f);
	fwrite_c(net->pSynapseNums, net->sTypeNum+1, f);

	for (size_t i=0; i<net->nTypeNum; i++) {
		saveType[net->pNTypes[i]](net->ppNeurons[i], net->pNeuronNums[i+1]-net->pNeuronNums[i], f);
	}
	for (size_t i=0; i<net->sTypeNum; i++) {
		saveType[net->pSTypes[i]](net->ppSynapses[i], net->pSynapseNums[i+1]-net->pSynapseNums[i], f);
	}

	for (size_t i=0; i<net->sTypeNum; i++) {
		saveConnection(net->ppConnections[i], f);
	}
	return 0;
}

GNetwork *loadGNetwork(FILE *f)
{
	size_t nTypeNum = 0, sTypeNum = 0;

	fread_c(&nTypeNum, 1, f);
	fread_c(&sTypeNum, 1, f);

	GNetwork * net = allocGNetwork(nTypeNum, sTypeNum);

	// fread(&(net->maxDelay), sizeof(size_t), 1, f);
	// fread(&(net->minDelay), sizeof(size_t), 1, f);

	fread_c(net->pNTypes, net->nTypeNum, f);
	fread_c(net->pSTypes, net->sTypeNum, f);
	fread_c(net->pNeuronNums, net->nTypeNum+1, f);
	fread_c(net->pSynapseNums, net->sTypeNum+1, f);

	for (size_t i=0; i<net->nTypeNum; i++) {
		net->ppNeurons[i] = loadType[net->pNTypes[i]](net->pNeuronNums[i+1]-net->pNeuronNums[i], f);
	}
	for (size_t i=0; i<net->sTypeNum; i++) {
		net->ppSynapses[i] = loadType[net->pSTypes[i]](net->pSynapseNums[i+1]-net->pSynapseNums[i], f);
	}

	net->ppConnections = malloc_c<Connection*>(sTypeNum);
	for (size_t i=0; i<net->sTypeNum; i++) {
		net->ppConnections[i] = loadConnection(f);
	}

	return net;
}

bool compareGNetwork(GNetwork *n1, GNetwork *n2)
{
	bool equal = true;
	equal = (n1->nTypeNum == n2->nTypeNum) && equal;
	equal = (n1->sTypeNum == n2->sTypeNum) && equal;

	equal = equal && isEqualArray(n1->pNTypes, n2->pNTypes, n1->nTypeNum);
	equal = equal && isEqualArray(n1->pSTypes, n2->pSTypes, n1->sTypeNum);
	equal = equal && isEqualArray(n1->pNeuronNums, n2->pNeuronNums, n1->nTypeNum+1);
	equal = equal && isEqualArray(n1->pSynapseNums, n2->pSynapseNums, n1->sTypeNum+1);

	for (size_t i=0; i<n1->nTypeNum; i++) {
		equal = isEqualType[n1->pNTypes[i]](n1->ppNeurons[i], n2->ppNeurons[i], n1->pNeuronNums[i+1]-n1->pNeuronNums[i], NULL, NULL) && equal;
	}
	for (size_t i=0; i<n1->sTypeNum; i++) {
		equal = isEqualType[n1->pSTypes[i]](n1->ppSynapses[i], n2->ppSynapses[i], n1->pSynapseNums[i+1]-n1->pSynapseNums[i], n1->ppConnections[i]->pSidMap, n2->ppConnections[i]->pSidMap) && equal;
	}

	for (size_t i=0; i<n1->sTypeNum; i++) {
		equal = equal && isEqualConnection(n1->ppConnections[i], n2->ppConnections[i]);
	}
	
	return equal;
}

int copyGNetwork(GNetwork *dNet, GNetwork *sNet, int rank, int rankSize)
{
	dNet->ppNeurons = sNet->ppNeurons;
	dNet->ppSynapses = sNet->ppSynapses;

	for (size_t i=0; i<dNet->nTypeNum; i++) {
		//size_t size = dNet->neuronNums[i+1] - dNet->neuronNums[i];
		//dNet->nOffsets[i] = 0;
		//Copy neurons
		//copyType[dNet->nTypes[i]](dNet, allNet, i, size);
	}
	for (size_t i=0; i<dNet->sTypeNum; i++) {
		//size_t size = dNet->neuronNums[i+1] - dNet->neuronNums[i];
		//dNet->sOffsets[i] = 0;
		//Copy synapses
		//copyType[network->sTypes[i]](network, allNet, i, size);
	}
	
	return 0;
}


int sendGNetwork(GNetwork *network, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(network, sizeof(GNetwork), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pNTypes, sizeof(Type)*(network->nTypeNum), MPI_UNSIGNED_CHAR, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pSTypes, sizeof(Type)*(network->sTypeNum), MPI_UNSIGNED_CHAR, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(network->pNeuronNums, network->nTypeNum+1, MPI_SIZE_T, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pSynapseNums, network->sTypeNum+1, MPI_SIZE_T, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	int tag_t = tag+5;
	for(size_t i=0; i<network->nTypeNum; i++) {
		sendType[network->pNTypes[i]](network->ppNeurons[i], dest, tag_t, comm);
		tag_t += TYPE_TAG;
	}
	for(size_t i=0; i<network->sTypeNum; i++) {
		sendType[network->pSTypes[i]](network->ppSynapses[i], dest, tag_t, comm);
		tag_t += TYPE_TAG;
	}

	for(size_t i=0; i<network->sTypeNum; i++) {
		ret = sendConnection(network->ppConnections[i], dest, tag_t, comm);
		tag_t += CONN_TAG;
		assert(ret == MPI_SUCCESS);
	}

	return ret;
}

GNetwork * recvGNetwork(int src, int tag, MPI_Comm comm) 
{
	GNetwork *net = (GNetwork*)malloc(sizeof(GNetwork));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(net, sizeof(GNetwork), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	net->pNTypes = (Type *)malloc(sizeof(Type) * (net->nTypeNum));
	net->pSTypes = (Type *)malloc(sizeof(Type) * (net->sTypeNum));

	ret = MPI_Recv(net->pNTypes, sizeof(Type)*(net->nTypeNum), MPI_UNSIGNED_CHAR, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pSTypes, sizeof(Type)*(net->sTypeNum), MPI_UNSIGNED_CHAR, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	net->pNeuronNums = (size_t *)malloc(sizeof(size_t) * (net->nTypeNum + 1));
	net->pSynapseNums = (size_t *)malloc(sizeof(size_t) * (net->sTypeNum + 1));

	ret = MPI_Recv(net->pNeuronNums, net->nTypeNum+1, MPI_SIZE_T, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pSynapseNums, net->sTypeNum+1, MPI_SIZE_T, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);

	net->ppNeurons = (void **)malloc(sizeof(void *) * (net->nTypeNum));
	net->ppSynapses = (void **)malloc(sizeof(void *) * (net->sTypeNum));

	int tag_t = tag+5;
	for(size_t i=0; i<net->nTypeNum; i++) {
		net->ppNeurons[i] = recvType[net->pNTypes[i]](src, tag_t, comm);
		tag_t += TYPE_TAG;
	}
	for(size_t i=0; i<net->sTypeNum; i++) {
		net->ppSynapses[i] = recvType[net->pSTypes[i]](src, tag_t, comm);
		tag_t += TYPE_TAG;
	}

	net->ppConnections = (Connection **)malloc(sizeof(Connection*)*(net->sTypeNum));
	for(size_t i=0; i<net->sTypeNum; i++) {
		net->ppConnections[i] = recvConnection(src, tag_t, comm);
		tag_t += CONN_TAG;
	}
	return net;
}

// bool isEqualGNetwork(GNetwork *n1, GNetwork *n2)
// {
// 	bool ret = true;
// 	ret = ret && (n1->nTypeNum == n2->nTypeNum);
// 	ret = ret && (n1->sTypeNum == n2->sTypeNum);
// 	ret = ret && isEqualArray(n1->pNTypes, n2->pNTypes, n1->nTypeNum);
// 	ret = ret && isEqualArray(n1->pSTypes, n2->pSTypes, n1->sTypeNum);
// 	ret = ret && isEqualArray(n1->pNeuronNums, n2->pNeuronNums, n1->nTypeNum+1);
// 	ret = ret && isEqualArray(n1->pSynapseNums, n2->pSynapseNums, n1->sTypeNum+1);
// 
// 	for (size_t i=0; i<n1->nTypeNum; i++) {
// 		ret = ret && isEqualType[n1->pNTypes[i]](n1->ppNeurons[i], n2->ppNeurons[i], n1->pNeuronNums[i+1]-n1->pNeuronNums[i]);
// 	}
// 	for (size_t i=0; i<n2->sTypeNum; i++) {
// 		ret = ret && isEqualType[n1->pSTypes[i]](n1->ppSynapses[i], n2->ppSynapses[i], n1->pSynapseNums[i+1]-n1->pSynapseNums[i]);
// 	}
// 
// 	for (size_t i=0; i<n2->sTypeNum; i++) {
// 		ret = ret && isEqualConnection(n1->ppConnections[i], n2->ppConnections[i]);
// 	}
// 
// 	return ret;
// }

// int printNetwork(GNetwork *net, int rank)
// {
// 	printf("NETWORK PRINT START...\n");
// 	// printf("Server: %d, nTypeNum: %d, sTypeNum: %d, maxDelay: %d, minDelay: %d\n", rank, net->nTypeNum, net->sTypeNum, net->maxDelay, net->minDelay);
// 
// 	printf("NTypes:");
// 	for(int i=0; i<net->nTypeNum; i++) {
// 		printf("%d ", net->nTypes[i]);
// 	}
// 	printf("\n");
// 	printf("STypes:");
// 	for(int i=0; i<net->sTypeNum; i++) {
// 		printf("%d ", net->sTypes[i]);
// 	}
// 	printf("\n");
// 
// 	printf("NNum:");
// 	for(int i=0; i<net->nTypeNum+1; i++) {
// 		printf("%d ", net->neuronNums[i]);
// 	}
// 	printf("\n");
// 	printf("SNum:");
// 	for(int i=0; i<net->sTypeNum+1; i++) {
// 		printf("%d ", net->synapseNums[i]);
// 	}
// 	printf("\n");
// 
// 	printf("Neurons:");
// 	for(int i=0; i<net->nTypeNum; i++) {
// 		printf("%d: %p\n", i, net->pNeurons[i]);
// 		//for (int i=0; i<(net->neuronNums[i+1]-net->neuronNums[i]); i++){
// 		//}
// 	}
// 	printf("\n");
// 	printf("Synapses:");
// 	for(int i=0; i<net->sTypeNum; i++) {
// 		printf("%d: %p\n", i, net->pSynapses[i]);
// 	}
// 	printf("\n");
// 
// 	printf("NETWORK PRINT END...\n");
// 
// 	return 0;
// }
