
#include "../../gpu_utils/runtime.h"

#include "StaticData.h"


__global__ void update_dense_static_hit(Connection *connection, StaticData *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int time)
{
#if 0
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int delayLength = connection->maxDelay - connection->minDelay + 1;
	for (int delta_t = 0; delta_t<delayLength; delta_t++) {
		int block_idx = blockIdx.x;
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);
		int firedSize = firedTableSizes[time_idx];
		if (firedSize > 0) {
			int num_per_block = (firedSize + gridDim.x - 1)/gridDim.x;
			int block_nums_minus_1 = (firedSize + num_per_block - 1) / num_per_block - 1;

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
				int startLoc = connection->pDelayStart[delta_t + nid * delayLength];
				int synapseNum = connection->pDelayNum[delta_t + nid * delayLength];
				if (threadIdx.x == 0) {
					gLayerInput[nid]++;
				}
				for (int j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					int sid = j+startLoc;
					real weight = data->pWeight[sid];
					if (weight >= 0) {
						atomicAdd(&(currentE[data->pDst[sid]]), weight);
					} else {
						atomicAdd(&(currentI[data->pDst[sid]]), weight);
					}
				}
			}
		}
		__syncthreads();
	}
#else
	int delayLength = connection->maxDelay - connection->minDelay + 1;
	for (int delta_t = 0; delta_t<delayLength; delta_t++) {
		int block_idx = blockIdx.x;
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);
		int firedSize = firedTableSizes[time_idx];
		if (firedSize > 0) {
			int num_per_block = (firedSize + gridDim.x - 1)/gridDim.x;
			int npb_minus_1 = num_per_block - 1;
			int full_offset = firedSize - npb_minus_1 * gridDim.x;

			int fired_size_block = 0;
			int nid_offset = 0;
			if (block_idx < full_offset) {
				fired_size_block = num_per_block;
				nid_offset = time_idx * gFiredTableCap + block_idx * num_per_block;
			} else if (block_idx < gridDim.x) {
				fired_size_block = npb_minus_1;
				nid_offset = time_idx * gFiredTableCap + block_idx * npb_minus_1 + full_offset;
			} else {
				fired_size_block = 0;
				nid_offset = 0;
			}

			for (int idx = 0; idx < fired_size_block; idx++) {
#ifdef DEBUG
				if (nid_offset + idx > firedSize + time_idx * gFiredTableCap) {
					printf("over flow %d + %d >= %d, from (%d, %d)\n", nid_offset, idx, firedSize + time_idx * gFiredTableCap, blockIdx.x, threadIdx.x);
					printf("%d %d %d %d %d %d %d\n", firedSize, gridDim.x, time_idx, gFiredTableCap, block_idx, npb_minus_1, full_offset);
				}
#endif
				int nid = firedTable[nid_offset + idx];
				int startLoc = connection->pDelayStart[delta_t + nid * delayLength];
				int synapseNum = connection->pDelayNum[delta_t + nid * delayLength];
				if (threadIdx.x == 0) {
					gLayerInput[nid]++;
				}
				for (int j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					int sid = j+startLoc;
					real weight = data->pWeight[sid];
					if (weight >= 0) {
						atomicAdd(&(currentE[connection->dst[connection->pSidMap[sid]]]), weight);
					} else {
						atomicAdd(&(currentI[connection->dst[connection->pSidMap[sid]]]), weight);
					}
				}
			}
		}
		__syncthreads();
	}
#endif
}

void cudaUpdateStatic(Connection * connection, void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int time, BlockSize *pSize)
{
	//update_static_hit<<<pSize->gridSize, pSize->blockSize>>>((StaticData*)data, num, start_id);
	//reset_active_synapse<<<1, 1>>>();
	update_dense_static_hit<<<pSize->gridSize, pSize->blockSize>>>((Connection *)connection,  (StaticData *)data, currentE, currentI, firedTable, firedTableSizes, num, start_id, time);

}

