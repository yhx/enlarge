
#ifndef STATICDATA_H
#define STATICDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct StaticData {
	bool is_view;
	size_t num;

	// int *pDst;
	real *pWeight;
};


size_t getStaticSize();
void *mallocStatic();
void *allocStatic(size_t num);
int allocStaticPara(void *pCPU, size_t num);
int freeStatic(void *pCPU);
int freeStaticPara(void *pCPU);
void updateStatic(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveStatic(void *pCPU, size_t num, const string &path);
void *loadStatic(size_t num, const string &path);
bool isEqualStatic(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);

int shuffleStatic(void *p, uinteger_t *shuffle, size_t num);

void *cudaMallocStatic();
void *cudaAllocStatic(void *pCPU, size_t num);
void *cudaAllocStaticPara(void *pCPU, size_t num);
int cudaFreeStatic(void *pGPU);
int cudaFreeStaticPara(void *pGPU);
int cudaFetchStatic(void *pCPU, void *pGPU, size_t num);
int cudaStaticParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaStaticParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateStatic(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);

int sendStatic(void *data, int dest, int tag, MPI_Comm comm);
void * recvStatic(int src, int tag, MPI_Comm comm);

#endif /* STATICDATA_H */
