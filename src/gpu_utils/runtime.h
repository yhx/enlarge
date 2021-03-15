/* This header file is writen by qp11
 * usually just for fun
 * Tue December 15 2015
 */
#ifndef RUNTIME_H
#define RUNTIME_H

#include "curand_kernel.h"
#include "curand.h"


#include "../utils/type.h"
#include "../utils/constant.h"
#include "../utils/BlockSize.h"
#include "../net/GNetwork.h"
#include "../net/Connection.h"

// Constant
// extern __constant__ int MAX_DELAY;
// extern __constant__ int gTimeTableCap;
extern __constant__ size_t gFiredTableCap;
// extern __constant__ int gSynapsesTableCap;
extern __constant__ real DT;

// Variable
// extern __device__ int gCurrentIdx;
// extern __device__ int gCurrentCycle;
// extern __device__ int gFiredTableSize;
// extern __device__ int gSynapsesActiveTableSize;

// #define gMin   0
// #define gMax 0.01

#define G_MAX -100
#define G_MIN 100
// Arrays
//extern __device__ int *gTimeTable;

// Neuron Arrays
// extern __device__ real *gNeuronInput;
// extern __device__ real *gNeuronInput_I;

// Neuron Tables
// extern __device__ int *gFiredTable;
// extern __device__ int *gFiredTableSizes;
// extern __device__ int *gActiveTable;
// extern __device__ int gActiveTableSize;

// Synapse Tables
//extern __device__ int *gSynapsesActiveTable;
//extern __device__ int *gSynapsesLogTable;

// Log Arrays
extern __device__ uinteger_t *gLayerInput;
extern __device__ real *gXInput;
extern __device__ int *gFireCount;

// Connection
// extern __device__ Connection *gConnection;


__global__ void init_connection(Connection *pConnection);

// __global__ void update_time(Connection *conn, int time, int *firedTableSizes);
__global__ void update_time(uinteger_t *firedTableSizes, int max_delay, int time);

__global__ void curand_setup_kernel(curandState *state, int num);

// __global__ void init_log_buffers(int * layer_input, real * x_input, int * fire_count);
	
// __global__ void init_buffers(/*int *c_gTimeTable,*/ real *c_gNeuronInput, real *c_gNeuronInput_I, int *c_gFiredTable, int *c_gFiredTableSizes, int *c_gActiveTable, int *c_gSynapsesActiveTable, int *c_gSynapsesLogTable);

__global__ void reset_active_synapse();

__global__ void cudaUpdateFTS(int * firedTableSizes, int num, int idx);

__global__ void cudaAddCrossNeurons(Connection *conn, int *firedTable, int *firedTableSizes, int *ids, int num, int time);

__global__ void cudaDeliverNeurons(int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, int *global_cross_data, int *fired_n_num, int max_delay, int node_num, int time);

__device__ real _clip(real a, real min, real max);

template<typename T>
__device__ int commit2globalTable(T *shared_buf, volatile size_t size, T *global_buf, T * global_size, size_t offset);

BlockSize * getBlockSize(int nSize, int sSize);


#endif /* RUNTIME_H */

