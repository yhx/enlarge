
#ifndef LIFDATA_H
#define LIFDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../utils/type.h"
#include "../../utils/BlockSize.h"

struct LIFData {
	int num;

	int *pRefracTime;
	int *pRefracStep;

	real *pI_e;
	real *pV_i;
	real *pCe;
	real *pV_reset;
	real *pV_tmp;
	real *pI_i;
	real *pV_thresh;
	real *pCi;
	real *pV_m;
	real *pC_e;
	real *pC_m;
	real *pC_i;
	// real *pV_e;
	bool is_view;
};


size_t getLIFSize();
void *mallocLIF();
void *allocLIF(int num);
int allocLIFPara(void *pCPU, int num);
int freeLIF(void *pCPU);
int freeLIFPara(void *pCPU);
void updateLIF(Connection *, void *, real *, real *, int *, int*, int, int, int, int);
int saveLIF(void *pCPU, int num, FILE *f);
void *loadLIF(int num, FILE *f);
bool isEqualLIF(void *p1, void *p2, int num);
int copyLIF(void *src, size_t s_off, void *dst, size_t d_off);

void *cudaMallocLIF();
void *cudaAllocLIF(void *pCPU, int num);
void *cudaAllocLIFPara(void *pCPU, int num);
int cudaFreeLIF(void *pGPU);
int cudaFreeLIFPara(void *pGPU);
int cudaFetchLIF(void *pCPU, void *pGPU, int num);
int cudaLIFParaToGPU(void *pCPU, void *pGPU, int num);
int cudaLIFParaFromGPU(void *pCPU, void *pGPU, int num);
void cudaUpdateLIF(Connection *conn, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int t, BlockSize *pSize);

int sendLIF(void *data, int dest, int tag, MPI_Comm comm);
void * recvLIF(int src, int tag, MPI_Comm comm);

#endif /* LIFDATA_H */
