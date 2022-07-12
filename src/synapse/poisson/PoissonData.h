
#ifndef POISSONDATA_H
#define POISSONDATA_H

#include <stdio.h>
#include <curand_kernel.h>
#include <curand.h>
#include <random>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../base/type.h"
#include "../../utils/BlockSize.h"
#include "nccl.h"

/**
 * PoissonData is used to manage data of poisson synapses.
 * It contains synapse number `num` and weights of each 
 * synapses `pWeight`. 
 **/
struct PoissonData {
	bool is_view;
	size_t num;

	// int *pDst;
	real *pWeight;
	real *pMean; 	// 均值
	curandState *pState;	// poisson state for CUDA
	std::poisson_distribution<int> *pPoissonGenerator;
	std::mt19937 *pGenerator;
};

/**
 * Functions related with CPU. They are implemented in 
 * PoissonData.cpp
 **/
size_t getPoissonSize();
void *mallocPoisson();
void *allocPoisson(size_t num);
int allocPoissonPara(void *pCPU, size_t num);
int freePoisson(void *pCPU);
int freePoissonPara(void *pCPU);
void updatePoisson(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int savePoisson(void *pCPU, size_t num, const string &path);
void *loadPoisson(size_t num, const string &path);
bool isEqualPoisson(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);

int shufflePoisson(void *p, uinteger_t *shuffle, size_t num);

/**
 * Funcions headed with cuda are related to GPU manipulations.
 * They are implemented in PoissonData.cu (mainly for memory
 * manipulations) and PoissonData.kernel.cu (mainly for computation).
 **/ 
void *cudaMallocPoisson();
void *cudaAllocPoisson(void *pCPU, size_t num);
void *cudaAllocPoissonPara(void *pCPU, size_t num);
int cudaFreePoisson(void *pGPU);
int cudaFreePoissonPara(void *pGPU);
int cudaFetchPoisson(void *pCPU, void *pGPU, size_t num);
int cudaPoissonParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaPoissonParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdatePoisson(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize, cudaStream_t cuda_stream);
void cudaUpdatePoisson(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);

/**
 * Functions related with MPI. 
 **/
int sendPoisson(void *data, int dest, int tag, MPI_Comm comm);
void * recvPoisson(int src, int tag, MPI_Comm comm);

#endif /* PoissonDATA_H */
