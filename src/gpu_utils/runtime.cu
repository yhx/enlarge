
#include <assert.h>

#include "../utils/utils.h"
// #include "../neuron/array/array.h"
// #include "../synapse/static/static.h"
// #include "../neuron/constant/constants.h"
// #include "../neuron/decide/decide.h"
// #include "../neuron/fft/fft.h"
// #include "../neuron/max/max.h"
// #include "../neuron/mem/mem.h"
// #include "../neuron/poisson/poisson.h"
// #include "../neuron/tj/tj.h"

#include "../../msg_utils/helper/helper_gpu.h"
#include "runtime.h"
// #include "gpu_func.h"



//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//#else
//__device__ double atomicAdd(double* address, double val)
//{
//	unsigned long long int* address_as_ull = (unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,
//				__double_as_longlong(val + __longlong_as_double(assumed)));
//	} while (assumed != old);
//	return __longlong_as_double(old);
//}
//#endif


// __constant__ int MAX_DELAY;
__constant__ int gTimeTableCap;
__constant__ size_t gFiredTableCap;
// __constant__ int gSynapsesTableCap;
__constant__ real DT;

// Variable
// __device__ int gCurrentIdx;
// __device__ int gCurrentCycle;
// __device__ int gFiredTableSize;
// __device__ int gSynapsesActiveTableSize;

// Arrays
//__device__ int *gTimeTable;

// Neuron Arrays
// __device__ real *gNeuronInput;
// __device__ real *gNeuronInput_I;

// Neuron Tables
// __device__ int *gFiredTable;
// __device__ int *gFiredTableSizes;
__device__ int *gActiveTable;
__device__ int gActiveTableSize;

// Synapse Tables
//__device__ int *gSynapsesActiveTable;
//__device__ int *gSynapsesLogTable;

// Log Arrays
__device__ uinteger_t *gLayerInput;
__device__ real *gXInput;
__device__ int *gFireCount;

// Connection
// __device__ N2SConnection *gConnection;


__device__ real _clip(real a, real min, real max)
{
	// real t = (a<min) ? min : a;
	// t = (t>max) ? max : t;
	// return t;

	if (a < min) {
		return min;
	} else if (a > max) {
		return max;
	} else {
		return a;
	}
}


__global__ void update_time(uinteger_t *firedTableSizes, int max_delay, int time)
{
	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
		// gCurrentCycle = gCurrentCycle + 1;
		// gCurrentIdx = (gCurrentIdx +1)%(MAX_DELAY + 1);
		int currentIdx = time % (max_delay + 1);
		gActiveTableSize = 0;
		firedTableSizes[currentIdx] = 0;
		// gSynapsesActiveTableSize = 0;
	}
	__syncthreads();
}
// 
// __global__ void init_time(int gCurrentCycle)
// {
// 	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
// 		//gTimeTable[gCurrentIdx] = simTime;
// 		gCurrentCycle = gCurrentCycle;
// 		gCurrentIdx = (gCurrentCycle)%(MAX_DELAY + 1);
// 		gActiveTableSize = 0;
// 		gFiredTableSizes[gCurrentIdx] = 0;
// 		gSynapsesActiveTableSize = 0;
// 	}
// 	__syncthreads();
// }

// __global__ void reset_active_synapse()
// {
// 	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
// 		gSynapsesActiveTableSize = 0;
// 	}
// 	__syncthreads();
// 
// }

__global__ void curand_setup_kernel(curandState *state, int num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < num) {
		curand_init(1234, id, 0, &state[id]); 
	}
}

__global__ void cudaUpdateFTS(int *firedTableSizes, int num, int idx)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		firedTableSizes[idx] += num;
	}
}

__global__ void cudaAddCrossNeurons(uinteger_t *firedTable, uinteger_t *firedTableSizes, uinteger_t *ids, size_t num, int max_delay, int time)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int delayIdx = time % (connection->maxDelay-connection->minDelay+1);
	int delayIdx = time % (max_delay+1);
	if (tid < num) {
		firedTable[gFiredTableCap*delayIdx + firedTableSizes[delayIdx] + tid] = ids[tid];
	}
	__syncthreads();

	if (tid == 0) {
		firedTableSizes[delayIdx] += num;
	}
}

__global__ void cudaDeliverNeurons(uinteger_t *firedTable, uinteger_t *firedTableSizes, integer_t *idx2index, integer_t *crossnode_index2idx, uinteger_t *global_cross_data, uinteger_t *fired_n_num, int max_delay, int node_num, int time)
{
	__shared__ uinteger_t cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile uinteger_t cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int delayIdx = time % (conn->maxDelay-conn->minDelay+1);
	int delayIdx = time % (max_delay+1);
	// int delayIdx = time % (conn->maxDelay+1);
	uinteger_t fired_size = firedTableSizes[delayIdx];
	for (int node = 0; node < node_num; node++) {
		for (uinteger_t i = 0; i < fired_size; i += blockDim.x * gridDim.x) {
			uinteger_t idx = i + tid;
			if (idx < fired_size) {
				uinteger_t nid = firedTable[gFiredTableCap*delayIdx + idx];
				integer_t tmp = idx2index[nid];
				if (tmp >= 0) {
					integer_t map_nid = crossnode_index2idx[tmp*node_num + node];
					if (map_nid >= 0) {
						size_t test_loc = atomicAdd((uinteger_t *)&cross_cnt, 1);
						if (test_loc < MAX_BLOCK_SIZE) {
							cross_neuron_id[test_loc] = map_nid;
						}
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				commit2globalTable(cross_neuron_id, cross_cnt, global_cross_data, &(fired_n_num[node]), static_cast<uinteger_t>(gFiredTableCap*node));
				if (threadIdx.x == 0) {
					cross_cnt = 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// __global__ void init_connection(N2SConnection *pConnection)
// {
// 	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
// 		gConnection = pConnection;
// 	}
// }

BlockSize * getBlockSize(int nSize, int sSize)
{
	BlockSize *ret = (BlockSize*)malloc(sizeof(BlockSize)*TYPESIZE);
	memset(ret, 0, sizeof(BlockSize)*TYPESIZE);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Array].minGridSize), &(ret[Array].blockSize), update_array_neuron, 0, nSize); 
	// ret[Array].gridSize = (upzero_else_set_one(nSize) + (ret[Array].blockSize) - 1) / (ret[Array].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[LIF].minGridSize), &(ret[LIF].blockSize), update_lif_neuron, 0, nSize); 
	ret[LIF].blockSize = 128;
	ret[LIF].gridSize = (upzero_else_set_one(nSize) + (ret[LIF].blockSize) - 1) / (ret[LIF].blockSize);

	ret[IAF].blockSize = 128;
	ret[IAF].gridSize = (upzero_else_set_one(nSize) + (ret[IAF].blockSize) - 1) / (ret[IAF].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Constant].minGridSize), &(ret[Constant].blockSize), update_constant_neuron, 0, nSize); 
	// ret[Constant].gridSize = (upzero_else_set_one(nSize) + (ret[Constant].blockSize) - 1) / (ret[Constant].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Poisson].minGridSize), &(ret[Poisson].blockSize), update_poisson_neuron, 0, nSize); 
	// ret[Poisson].gridSize = (upzero_else_set_one(nSize) + (ret[Poisson].blockSize) - 1) / (ret[Poisson].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Decide].minGridSize), &(ret[Decide].blockSize), update_max_neuron, 0, nSize); 
	// ret[Decide].gridSize = (upzero_else_set_one(nSize) + (ret[Decide].blockSize) - 1) / (ret[Decide].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[FFT].minGridSize), &(ret[FFT].blockSize), update_fft_neuron, 0, nSize); 
	// ret[FFT].gridSize = (upzero_else_set_one(nSize) + (ret[FFT].blockSize) - 1) / (ret[FFT].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Mem].minGridSize), &(ret[Mem].blockSize), update_mem_neuron, 0, nSize); 
	// ret[Mem].gridSize = (upzero_else_set_one(nSize) + (ret[Mem].blockSize) - 1) / (ret[Mem].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Max].minGridSize), &(ret[Max].blockSize), update_max_neuron, 0, nSize); 
	// ret[Max].gridSize = (upzero_else_set_one(nSize) + (ret[Max].blockSize) - 1) / (ret[Max].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[TJ].minGridSize), &(ret[TJ].blockSize), update_tj_neuron, 0, nSize); 
	// ret[TJ].gridSize = (upzero_else_set_one(nSize) + (ret[TJ].blockSize) - 1) / (ret[TJ].blockSize);

	// cudaOccupancyMaxPotentialBlockSize(&(ret[Static].minGridSize), &(ret[Static].blockSize), update_static_hit, 0, sSize); 
	ret[Static].blockSize = 128;
	ret[Static].gridSize = (upzero_else_set_one(nSize) + (ret[Static].blockSize) - 1) / (ret[Static].blockSize);
	// ret[Static].gridSize = (upzero_else_set_one(nSize) + (ret[Static].blockSize) - 1) / (16);

	ret[Poisson].blockSize = 128;
	ret[Poisson].gridSize = (upzero_else_set_one(nSize) + (ret[Poisson].blockSize) - 1) / (ret[Poisson].blockSize);

	return ret;
}
