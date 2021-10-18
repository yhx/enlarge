
#ifndef PoissonDATA_H
#define PoissonDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct PoissonData {
	bool is_view;
	size_t num;

	// int *pDst;
	real *pWeight;
};


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

void *cudaMallocPoisson();
void *cudaAllocPoisson(void *pCPU, size_t num);
void *cudaAllocPoissonPara(void *pCPU, size_t num);
int cudaFreePoisson(void *pGPU);
int cudaFreePoissonPara(void *pGPU);
int cudaFetchPoisson(void *pCPU, void *pGPU, size_t num);
int cudaPoissonParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaPoissonParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdatePoisson(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);

int sendPoisson(void *data, int dest, int tag, MPI_Comm comm);
void * recvPoisson(int src, int tag, MPI_Comm comm);

#endif /* PoissonDATA_H */
