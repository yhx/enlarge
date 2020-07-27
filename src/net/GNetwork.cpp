/* This program is writen by qp09.
 * usually just for fun.
 * 一 四月 18 2016
 */

#include <mpi.h>
#include <assert.h>
#include <iostream>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "../utils/FileOp.h"

#include "GNetwork.h"

GNetwork * deepcopyNetwork(GNetwork * net) {
	GNetwork * ret = allocNetwork(net->nTypeNum, net->sTypeNum);

	for (int i=0; i<net->nTypeNum; i++) {
		ret->pNTypes[i] = net->pNTypes[i];
		ret->pNeuronNums[i] = net->pNeuronNums[i];
		ret->ppNeurons[i] = allocType[net->pNTypes[i]](net->pNeuronNums[i+1]-net->pNeuronNums[i]);
	}
	ret->pNeuronNums[net->nTypeNum] = net->pNeuronNums[net->nTypeNum];

	for (int i=0; i<net->sTypeNum; i++) {
		ret->pSTypes[i] = net->pSTypes[i];
		ret->pSynapseNums[i] = net->pSynapseNums[i];
		ret->ppSynapses[i] = allocType[net->pSTypes[i]](net->pSynapseNums[i+1]-net->pSynapseNums[i]);
	}
	ret->pSynapseNums[net->sTypeNum] = net->pSynapseNums[net->sTypeNum];

	ret->pConnection = allocConnection(net->pConnection->nNum, net->pConnection->sNum, 
			net->pConnection->maxDelay, net->pConnection->minDelay);

	return ret;
}

GNetwork * allocNetwork(int nTypeNum, int sTypeNum) {
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

	ret->pNeuronNums = (int*)malloc(sizeof(int)*(nTypeNum + 1));
	assert(ret->pNeuronNums != NULL);
	ret->pSynapseNums = (int*)malloc(sizeof(int)*(sTypeNum + 1));
	assert(ret->pSynapseNums != NULL);

	ret->pNeuronNums[0] = 0;
	ret->pSynapseNums[0] = 0;

	ret->ppNeurons = (void **)malloc(sizeof(void*)*nTypeNum);
	assert(ret->ppNeurons != NULL);
	ret->ppSynapses = (void **)malloc(sizeof(void*)*sTypeNum);
	assert(ret->ppSynapses != NULL);

	ret->pConnection = NULL;

	return ret;
}


void freeGNetwork(GNetwork * network)
{
	for (int i=0; i<network->nTypeNum; i++) {
		freeType[network->pNTypes[i]](network->ppNeurons[i]);
	}
	free(network->ppNeurons);

	for (int i=0; i<network->sTypeNum; i++) {
		freeType[network->pSTypes[i]](network->ppSynapses[i]);
	}
	free(network->ppSynapses);

	freeConnection(network->pConnection);

	free(network->pNeuronNums);
	free(network->pSynapseNums);

	free(network->pNTypes);
	free(network->pSTypes);
}

int saveGNetwork(GNetwork *net, const char *filename)
{
	FILE * f = openFile(filename, "w+");

	fwrite(&(net->nTypeNum), sizeof(int), 1, f);
	fwrite(&(net->sTypeNum), sizeof(int), 1, f);
	// fwrite(&(net->maxDelay), sizeof(int), 1, f);
	// fwrite(&(net->minDelay), sizeof(int), 1, f);

	fwrite(net->pNTypes, sizeof(Type), net->nTypeNum, f);
	fwrite(net->pSTypes, sizeof(Type), net->sTypeNum, f);
	fwrite(net->pNeuronNums, sizeof(int), net->nTypeNum+1, f);
	fwrite(net->pSynapseNums, sizeof(int), net->sTypeNum+1, f);

	for (int i=0; i<net->nTypeNum; i++) {
		saveType[net->pNTypes[i]](net->ppNeurons[i], net->pNeuronNums[i+1]-net->pNeuronNums[i], f);
	}
	for (int i=0; i<net->sTypeNum; i++) {
		saveType[net->pSTypes[i]](net->ppSynapses[i], net->pSynapseNums[i+1]-net->pSynapseNums[i], f);
	}

	saveConnection(net->pConnection, f);

	closeFile(f);
	return 0;
}

GNetwork *loadGNetwork(const char *filename)
{
	int nTypeNum = 0, sTypeNum = 0;
	FILE * f = openFile(filename, "r");

	fread(&nTypeNum, sizeof(int), 1, f);
	fread(&sTypeNum, sizeof(int), 1, f);

	GNetwork * net = allocNetwork(nTypeNum, sTypeNum);

	// fread(&(net->maxDelay), sizeof(int), 1, f);
	// fread(&(net->minDelay), sizeof(int), 1, f);

	fread(net->pNTypes, sizeof(Type), net->nTypeNum, f);
	fread(net->pSTypes, sizeof(Type), net->sTypeNum, f);
	fread(net->pNeuronNums, sizeof(int), net->nTypeNum+1, f);
	fread(net->pSynapseNums, sizeof(int), net->sTypeNum+1, f);

	for (int i=0; i<net->nTypeNum; i++) {
		net->ppNeurons[i] = loadType[net->pNTypes[i]](net->pNeuronNums[i+1]-net->pNeuronNums[i], f);
	}
	for (int i=0; i<net->sTypeNum; i++) {
		net->ppSynapses[i] = loadType[net->pSTypes[i]](net->pSynapseNums[i+1]-net->pSynapseNums[i], f);
	}

	net->pConnection = loadConnection(f);

	closeFile(f);
	return net;
}

int copyNetwork(GNetwork *dNet, GNetwork *sNet, int rank, int rankSize)
{
	dNet->ppNeurons = sNet->ppNeurons;
	dNet->ppSynapses = sNet->ppSynapses;

	for (int i=0; i<dNet->nTypeNum; i++) {
		//int size = dNet->neuronNums[i+1] - dNet->neuronNums[i];
		//dNet->nOffsets[i] = 0;
		//Copy neurons
		//copyType[dNet->nTypes[i]](dNet, allNet, i, size);
	}
	for (int i=0; i<dNet->sTypeNum; i++) {
		//int size = dNet->neuronNums[i+1] - dNet->neuronNums[i];
		//dNet->sOffsets[i] = 0;
		//Copy synapses
		//copyType[network->sTypes[i]](network, allNet, i, size);
	}
	
	return 0;
}


int sendNetwork(GNetwork *network, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(network, sizeof(GNetwork), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pNTypes, sizeof(Type)*(network->nTypeNum), MPI_UNSIGNED_CHAR, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pSTypes, sizeof(Type)*(network->sTypeNum), MPI_UNSIGNED_CHAR, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(network->pNeuronNums, network->nTypeNum+1, MPI_INT, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(network->pSynapseNums, network->sTypeNum+1, MPI_INT, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	int tag_t = tag+5;
	for(int i=0; i<network->nTypeNum; i++) {
		sendType[network->pNTypes[i]](network->ppNeurons, dest, tag_t, comm);
		tag_t += TYPE_TAG;
	}
	for(int i=0; i<network->sTypeNum; i++) {
		sendType[network->pSTypes[i]](network->ppSynapses, dest, tag_t, comm);
		tag_t += TYPE_TAG;
	}

	ret = sendConnection(network->pConnection, dest, tag_t+1, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

GNetwork * mpiRecvNetwork(GNetwork *network, int src, int tag, MPI_Comm comm) 
{
	GNetwork *ret = (GNetwork*)malloc(sizeof(GNetwork));
	MPI_Status status;
	MPI_Recv(ret, sizeof(GNetwork), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->pNTypes = (Type *)malloc(sizeof(Type) * (ret->nTypeNum));
	ret->pSTypes = (Type *)malloc(sizeof(Type) * (ret->sTypeNum));

	MPI_Recv(ret->pNTypes, sizeof(Type)*(ret->nTypeNum), MPI_UNSIGNED_CHAR, src, tag+1, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	MPI_Recv(ret->pSTypes, sizeof(Type)*(ret->sTypeNum), MPI_UNSIGNED_CHAR, src, tag+2, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->pNeuronNums = (int *)malloc(sizeof(int) * (ret->nTypeNum + 1));
	ret->pSynapseNums = (int *)malloc(sizeof(int) * (ret->sTypeNum + 1));

	MPI_Recv(ret->pNeuronNums, ret->nTypeNum+1, MPI_INT, src, tag+3, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	MPI_Recv(ret->pSynapseNums, ret->sTypeNum+1, MPI_INT, src, tag+4, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->ppNeurons = (void **)malloc(sizeof(void *) * (ret->nTypeNum));
	ret->ppSynapses = (void **)malloc(sizeof(void *) * (ret->sTypeNum));

	int tag_t = tag+5;
	for(int i=0; i<network->nTypeNum; i++) {
		ret->ppNeuron[i] = recvType[network->pNTypes[i]](network->ppNeurons, src, tag_t, comm);
		tag_t += TYPE_TAG;
	}
	for(int i=0; i<network->sTypeNum; i++) {
		ret->ppSynapse[i] = recvType[network->pSTypes[i]](network->ppSynapses, src, tag_t, comm);
		tag_t += TYPE_TAG;
	}

	ret->pConnection = recvConnection(network->pConnection, dest, tag_t+1, comm);
	return ret;
}

bool isEqualNetwork(GNetwork *n1, GNetwork *n2)
{
	bool ret = true;
	ret = ret && (n1->nTypeNum == n2->nTypeNum);
	ret = ret && (n1->sTypeNum == n2->sTypeNum);
	ret = ret && isEqualArray(n1->pNTypes, n2->pNTypes, n1->nTypeNum);
	ret = ret && isEqualArray(n1->pSTypes, n2->pSTypes, n1->sTypeNum);
	ret = ret && isEqualArray(n1->pNeuronNums, n2->pNeuronNums, n1->nTypeNum+1);
	ret = ret && isEqualArray(n1->pSynapseNums, n2->pSynapseNums, n1->sTypeNum+1);

	for (int i=0; i<n1->nTypeNum; i++) {
		ret = ret && isEqualType[n1->pNTypes[i]](n1->ppNeurons[i], n2->ppNeurons[i], n1->pNeuronNums[i+1]-n1->pNeuronNums[i]);
	}
	for (int i=0; i<n2->sTypeNum; i++) {
		ret = ret && isEqualType[n1->pSTypes[i]](n1->ppSynapses[i], n2->ppSynapses[i], n1->pSynapseNums[i+1]-n1->pSynapseNums[i]);
	}

	ret = ret && isEqualConnection(n1->pConnection, n2->pConnection);

	return ret;
}

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
