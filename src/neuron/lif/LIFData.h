
#ifndef LIFDATA_H
#define LIFDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct LIFData {
	bool is_view;
	size_t num;

	int *pRefracTime;
	int *pRefracStep;

	real *pI_e;
	// real *pV_i;
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
};


size_t getLIFSize();
void *mallocLIF();
void *allocLIF(size_t num);
int allocLIFPara(void *pCPU, size_t num);
int freeLIF(void *pCPU);
int freeLIFPara(void *pCPU);
void updateLIF(Connection *, void *, real *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveLIF(void *pCPU, size_t num, FILE *f);
void *loadLIF(size_t num, FILE *f);
bool isEqualLIF(void *p1, void *p2, size_t num, size_t *shuffle1=NULL, size_t *shuffle2=NULL);
int copyLIF(void *src, size_t s_off, void *dst, size_t d_off);

void *cudaMallocLIF();
void *cudaAllocLIF(void *pCPU, size_t num);
void *cudaAllocLIFPara(void *pCPU, size_t num);
int cudaFreeLIF(void *pGPU);
int cudaFreeLIFPara(void *pGPU);
int cudaFetchLIF(void *pCPU, void *pGPU, size_t num);
int cudaLIFParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaLIFParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateLIF(Connection *conn, void *data, real *currentE, real *currentI, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);

int sendLIF(void *data, int dest, int tag, MPI_Comm comm);
void * recvLIF(int src, int tag, MPI_Comm comm);

#endif /* LIFDATA_H */
