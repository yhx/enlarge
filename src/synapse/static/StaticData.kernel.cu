
#include "../../gpu_utils/runtime.h"

#include "StaticData.h"


__global__ void update_dense_static_hit(Connection *connection, StaticData *data, real *currentE, real *currentI, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTabelCap, size_t num, size_t start_id, int time)
{
	int delayLength = connection->maxDelay - connection->minDelay + 1;
	for (int delta_t = 0; delta_t<delayLength; delta_t++) {
		int block_idx = blockIdx.x;
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);
	    uinteger_t firedSize = firedTableSizes[time_idx];
		if (firedSize > 0) {
			uinteger_t num_per_block = (firedSize + gridDim.x - 1)/gridDim.x;
			uinteger_t npb_minus_1 = num_per_block - 1;
			uinteger_t full_offset = firedSize - npb_minus_1 * gridDim.x;

			uinteger_t fired_size_block = 0;
			uinteger_t nid_offset = 0;
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

			for (uinteger_t idx = 0; idx < fired_size_block; idx++) {
#ifdef DEBUG
				if (nid_offset + idx > firedSize + time_idx * gFiredTableCap) {
					printf("over flow %lld + %ld >= %lld, from (%d, %d)\n", (long long)nid_offset, (long long)idx, (long long)(firedSize + time_idx * gFiredTableCap), blockIdx.x, threadIdx.x);
					printf("%lld %d %d %ld %d %lld %ld\n", (long long)firedSize, gridDim.x, time_idx, gFiredTableCap, block_idx, (long long)npb_minus_1, (long long)full_offset);
				}
#endif
				uinteger_t nid = firedTable[nid_offset + idx];
				uinteger_t startLoc = access_(connection->pDelayStart, delta_t, nid);
				uinteger_t synapseNum = access_(connection->pDelayNum, delta_t, nid);
				if (threadIdx.x == 0) {
					gLayerInput[nid]++;
				}
				for (uinteger_t j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					uinteger_t sid = j+startLoc;
					real weight = data->pWeight[connection->pSidMap[sid]];
					if (weight >= 0) {
						atomicAdd(&(currentE[connection->dst[sid]]), weight);
					} else {
						atomicAdd(&(currentI[connection->dst[sid]]), weight);
					}
				}
			}
		}
		__syncthreads();
	}
}

void cudaUpdateStatic(Connection * connection, void *data, real *currentE, real *currentI, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time, BlockSize *pSize)
{
	//update_static_hit<<<pSize->gridSize, pSize->blockSize>>>((StaticData*)data, num, start_id);
	//reset_active_synapse<<<1, 1>>>();
	update_dense_static_hit<<<pSize->gridSize, pSize->blockSize>>>((Connection *)connection,  (StaticData *)data, currentE, currentI, firedTable, firedTableSizes, firedTableCap, num, start_id, time);

}

