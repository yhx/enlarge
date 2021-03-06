
#include "../../gpu_utils/runtime_info.h"

// #include "GStaticSynapses.h"
#include "static.h"


__global__ void update_dense_static_hit(N2SConnection *connection, GStaticSynapses *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int time)
{
#define FAST_TEST 2
#if  FAST_TEST == 1
	__shared__ int fire_neuron_id[MAXBLOCKSIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int delta_t = 0; delta_t<MAX_DELAY; delta_t++) {
		int block_idx = blockIdx.x;
		int time_idx = (time+MAX_DELAY-delta_t)%(MAX_DELAY+1);
		int firedSize = firedTableSizes[time_idx];
		int block_nums_minus_1 = (firedSize - 1 + blockDim.x) / blockDim.x - 1;
		int grid_nums = (firedSize - 1 + blockDim.x*gridDim.x)/(blockDim.x * gridDim.x);
		int oid = tid;
		for (int idx = 0; idx < grid_nums; idx++) {
			if (oid < firedSize) {
				fire_neuron_id[threadIdx.x] = firedTable[time_idx*gFiredTableCap + oid];
			} else {
				fire_neuron_id[threadIdx.x] = -1;
			}
			oid += blockDim.x * gridDim.x;
			__syncthreads();

			int size = 0;
			if (block_idx == block_nums_minus_1) {
				size = firedSize - block_idx * blockDim.x;
			} else if (block_idx < block_nums_minus_1) {
				size = blockDim.x;
			} else {
				size = 0;
			}

			for (int i=0; i<size; i++) {
				int nid = fire_neuron_id[i];
				int start_loc = connection->delayStart[delta_t + nid * MAX_DELAY];
				int synapseNum = connection->delayNum[delta_t + nid * MAX_DELAY];
				gLayerInput[nid]++;
				for (int j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+start_loc];
					int sid = j+start_loc;
					real weight = data->p_weight[sid];
					if (weight >= 0) {
						atomicAdd(&(currentE[data->p_dst[sid]]), weight);
					} else {
						atomicAdd(&(currentI[data->p_dst[sid]]), weight);
					}
				}
			}
			block_idx += gridDim.x;
			__syncthreads();
		}
		__syncthreads();
	}
#elif FAST_TEST == 2
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int delta_t = 0; delta_t<MAX_DELAY; delta_t++) {
		int block_idx = blockIdx.x;
		int time_idx = (time + MAX_DELAY-delta_t)%(MAX_DELAY+1);
		int firedSize = firedTableSizes[time_idx];
		int num_per_block = (firedSize - 1)/gridDim.x + 1;
		int block_nums_minus_1 = (firedSize - 1) / num_per_block;

		int fired_size_block = 0;
		if (block_idx == block_nums_minus_1) {
			fired_size_block = firedSize - block_idx * num_per_block;
		} else if (block_idx < block_nums_minus_1) {
			fired_size_block = num_per_block;
		} else {
			fired_size_block = 0;
		}

		for (int idx = 0; idx < fired_size_block; idx++) {
			int nid = firedTable[time_idx*gFiredTableCap + (block_idx)*num_per_block + idx];
			int start_loc = connection->delayStart[delta_t + nid * MAX_DELAY];
			int synapseNum = connection->delayNum[delta_t + nid * MAX_DELAY];
			if (threadIdx.x == 0) {
				gLayerInput[nid]++;
			}
			for (int j=threadIdx.x; j<synapseNum; j += blockDim.x) {
				//int sid = connection->pSynapsesIdx[j+start_loc];
				int sid = j+start_loc;
				real weight = data->p_weight[sid];
				if (weight >= 0) {
					atomicAdd(&(currentE[data->p_dst[sid]]), weight);
				} else {
					atomicAdd(&(currentI[data->p_dst[sid]]), weight);
				}
			}
		}
		__syncthreads();
	}
#else
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int delta_t = 0; delta_t<MAX_DELAY; delta_t++) {
		int time_idx = (time+MAX_DELAY-delta_t)%(MAX_DELAY+1);
		int firedSize = firedTableSizes[time_idx];
		for (int idx = tid; idx < firedSize; idx += blockDim.x*gridDim.x) {
			int nid = firedTable[time_idx*gFiredTableCap + idx];
			int start_loc = connection->delayStart[delta_t + nid * MAX_DELAY];
			int synapseNum = connection->delayNum[delta_t + nid * MAX_DELAY];
			gLayerInput[nid]++;
			for (int i=0; i<synapseNum; i++) {
				//int sid = connection->pSynapsesIdx[i+start_loc];
				int sid = i+start_loc;
				real weight = data->p_weight[sid];
				if (weight >= 0) {
					atomicAdd(&(currentE[data->p_dst[sid]]), weight);
				} else {
					atomicAdd(&(currentI[data->p_dst[sid]]), weight);
				}
			}
		}
	}
#endif
}

void cudaUpdateStatic(void * connection, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int time, BlockSize *pSize)
{
	//update_static_hit<<<pSize->gridSize, pSize->blockSize>>>((GStaticSynapses*)data, num, start_id);
	//reset_active_synapse<<<1, 1>>>();
	update_dense_static_hit<<<pSize->gridSize, pSize->blockSize>>>((N2SConnection *)connection,  (GStaticSynapses *)data, currentE, currentI, firedTable, firedTableSizes, num, start_id, time);

}

