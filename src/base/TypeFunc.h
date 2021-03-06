/* This header file is writen by qp09
 * usually just for fun
 * Thu December 29 2016
 */
#ifndef TYPEFUNC_H
#define TYPEFUNC_H

#include <stdio.h>
#include <string>
#include "mpi.h"

#include "type.h"
#include "../net/Connection.h"
#include "../utils/BlockSize.h"

using std::string;

extern size_t (*getTypeSize[TYPESIZE])();
extern void* (*mallocType[TYPESIZE])();
extern void* (*allocType[TYPESIZE])(size_t num);
extern int (*freeType[TYPESIZE])(void *pCPU);
extern int (*allocTypePara[TYPESIZE])(void *pCPU, size_t num);
extern int (*freeTypePara[TYPESIZE])(void *pCPU);

extern int (*saveType[TYPESIZE])(void *pCPU, size_t num, const string &path);
extern void* (*loadType[TYPESIZE])(size_t num, const string &path);

// extern int (*addTypeConnection[TYPESIZE])(void *, int *);
extern void (*updateType[TYPESIZE])(Connection *, void *, real *, uinteger_t *, uinteger_t*,  size_t, size_t, size_t, int);

extern bool (*isEqualType[TYPESIZE])(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2);

extern int (*logRateNeuron[TYPESIZE])(void *p1, const char *name);

extern int (*shuffleSynapse[TYPESIZE])(void *p1, uinteger_t *shuffle1, size_t num);

extern void *(*cudaAllocType[TYPESIZE])(void *pCPU, size_t num);
// extern int (*cudaTypeToGPU[TYPESIZE])(void *pCPU, void *pGPU, int num);
extern int (*cudaFetchType[TYPESIZE])(void *pCPU, void *pGPU, size_t num);
extern int (*cudaFreeType[TYPESIZE])(void *);
// extern void (*cudaFindType[TYPESIZE])(void *, int, int);
// extern void (*cudaUpdateNeuron[TYPESIZE])(void *, real *, real *, int *, int*, int, int, int, BlockSize *);
// extern void (*cudaUpdateSynapse[TYPESIZE])(void *, void *, real *, real *, int *, int*, int, int, int, BlockSize *);
extern void (*cudaUpdateType[TYPESIZE])(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int, BlockSize *);

extern int (*cudaLogRateNeuron[TYPESIZE])(void *cpu, void *gpu, const char *name);

extern int (*sendType[TYPESIZE])(void *data, int dest, int tag, MPI_Comm comm);
extern void * (*recvType[TYPESIZE])(int src, int tag, MPI_Comm comm);

extern BlockSize * getBlockSize(int nSize, int sSize);

// extern Type * (*castType[TYPESIZE])(void *data);

extern real * (*getVNeuron[TYPESIZE])(void *data);

extern real * (*cudaGetVNeuron[TYPESIZE])(void *data);

#endif /* TYPEFUNC_H */
