
#ifndef IZHIKEVICHDATA_H
#define IZHIKEVICHDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../utils/type.h"
#include "../../utils/BlockSize.h"

struct IzhikevichData 
{
	real *pV;
	real *pU;
	real *pA;
	real *pB;
	real *pC;
	real *pD;

};


size_t getIzhikevichSize();
void *mallocIzhikevich();
void *allocIzhikevich(int num);
int allocIzhikevichPara(void *pCPU, int num);
int freeIzhikevich(void *pCPU);
int freeIzhikevichPara(void *pCPU);
int saveIzhikevich(void *pCPU, int num, FILE *f);
void updateIzhikevich(Connection *conn, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int firedTableCap, int num, int start_id, int t);
void *loadIzhikevich(int num, FILE *f);
bool isEqualIzhikevich(void *p1, void *p2, int num);

void *cudaMallocIzhikevich();
void *cudaAllocIzhikevich(void *pCPU, int num);
void *cudaAllocIzhikevichPara(void *pCPU, int num);
int cudaFreeIzhikevich(void *pGPU);
int cudaFreeIzhikevichPara(void *pGPU);
int cudaFetchIzhikevich(void *pCPU, void *pGPU, int num);
int cudaIzhikevichParaToGPU(void *pCPU, void *pGPU, int num);
int cudaIzhikevichParaFromGPU(void *pCPU, void *pGPU, int num);
void cudaUpdateIzhikevich(Connection *conn, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int t, BlockSize *pSize);

int sendIzhikevich(void *data, int dest, int tag, MPI_Comm comm);
void * recvIzhikevich(int src, int tag, MPI_Comm comm);

#endif /* IZHIKEVICHDATA_H */
