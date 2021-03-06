
#ifndef TRAUBMILESDATA_H
#define TRAUBMILESDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../utils/type.h"
#include "../../utils/BlockSize.h"

struct TraubMilesData 
{
	real *pGNa;
	real *pENa;
	real *pGK;
	real *pEK;
	real *pGl;
	real *pEl;
	real *pC;
	real *pV;
	real *pM;
	real *pH;
	real *pN;
	real *pDecayI;
	real *pDecayE;
	real *pTau;
	real *pEI;
	real *pEE;

};


size_t getTraubMilesSize();
void *mallocTraubMiles();
void *allocTraubMiles(int num);
int allocTraubMilesPara(void *pCPU, int num);
int freeTraubMiles(void *pCPU);
int freeTraubMilesPara(void *pCPU);
int saveTraubMiles(void *pCPU, int num, FILE *f);
void updateTraubMiles(Connection *conn, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int firedTableCap, int num, int start_id, int t);
void *loadTraubMiles(int num, FILE *f);
bool isEqualTraubMiles(void *p1, void *p2, int num);

void *cudaMallocTraubMiles();
void *cudaAllocTraubMiles(void *pCPU, int num);
void *cudaAllocTraubMilesPara(void *pCPU, int num);
int cudaFreeTraubMiles(void *pGPU);
int cudaFreeTraubMilesPara(void *pGPU);
int cudaFetchTraubMiles(void *pCPU, void *pGPU, int num);
int cudaTraubMilesParaToGPU(void *pCPU, void *pGPU, int num);
int cudaTraubMilesParaFromGPU(void *pCPU, void *pGPU, int num);
void cudaUpdateTraubMiles(Connection *conn, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int t, BlockSize *pSize);

int sendTraubMiles(void *data, int dest, int tag, MPI_Comm comm);
void * recvTraubMiles(int src, int tag, MPI_Comm comm);

#endif /* TRAUBMILESDATA_H */
