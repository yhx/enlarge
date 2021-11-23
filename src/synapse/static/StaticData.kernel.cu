
#include "../../gpu_utils/runtime.h"

#include "StaticData.h"


__global__ void update_dense_static_hit(Connection *connection, StaticData *data, real *buffer, const uinteger_t *firedTable, const uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
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
				nid_offset = time_idx * firedTableCap + block_idx * num_per_block;
			} else if (block_idx < gridDim.x) {
				fired_size_block = npb_minus_1;
				nid_offset = time_idx * firedTableCap + block_idx * npb_minus_1 + full_offset;
			} else {
				fired_size_block = 0;
				nid_offset = 0;
			}

			for (uinteger_t idx = 0; idx < fired_size_block; idx++) {
#ifdef DEBUG
				if (nid_offset + idx > firedSize + time_idx * firedTableCap) {
					printf("over flow %lld + %ld >= %lld, from (%d, %d)\n", (long long)nid_offset, (long long)idx, (long long)(firedSize + time_idx * firedTableCap), blockIdx.x, threadIdx.x);
					printf("%lld %d %d %ld %d %lld %ld\n", (long long)firedSize, gridDim.x, time_idx, firedTableCap, block_idx, (long long)npb_minus_1, (long long)full_offset);
				}
#endif
				uinteger_t nid = firedTable[nid_offset + idx];
				// uinteger_t startLoc = access_(connection->pDelayStart, delta_t, nid);
				// uinteger_t synapseNum = access_(connection->pDelayNum, delta_t, nid);
				uinteger_t startLoc = access_connection_(connection->pDelayStart, delta_t, connection->nNum, nid);
				uinteger_t synapseNum = access_connection_(connection->pDelayNum, delta_t, connection->nNum, nid);
				// if (threadIdx.x == 0) {
				// 	gLayerInput[nid]++;
				// }
				for (uinteger_t j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					uinteger_t sid = j+startLoc;
					real weight = data->pWeight[connection->pSidMap[sid]];
					atomicAdd(&(buffer[connection->dst[sid]]), weight);
				}
			}
		}
		__syncthreads();
	}
}

__global__ void update_denser_static_hit(Connection *connection, StaticData *data, real *buffer, const uinteger_t *firedTable, const uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
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
				nid_offset = time_idx * firedTableCap + block_idx * num_per_block;
			} else if (block_idx < gridDim.x) {
				fired_size_block = npb_minus_1;
				nid_offset = time_idx * firedTableCap + block_idx * npb_minus_1 + full_offset;
			} else {
				fired_size_block = 0;
				nid_offset = 0;
			}

			// for (uinteger_t idx = 0; idx < fired_size_block; idx++) {
			for (uinteger_t t=threadIdx.x; t<fired_size_block; t+=blockDim.x) {
#ifdef DEBUG
				if (nid_offset + t > firedSize + time_idx * firedTableCap) {
					printf("over flow %lld + %ld >= %lld, from (%d, %d)\n", (long long)nid_offset, (long long)t, (long long)(firedSize + time_idx * firedTableCap), blockIdx.x, threadIdx.x);
					printf("%lld %d %d %ld %d %lld %ld\n", (long long)firedSize, gridDim.x, time_idx, firedTableCap, block_idx, (long long)npb_minus_1, (long long)full_offset);
				}
#endif
				uinteger_t nid = firedTable[nid_offset + t];
				// uinteger_t startLoc = access_(connection->pDelayStart, delta_t, nid);
				// uinteger_t synapseNum = access_(connection->pDelayNum, delta_t, nid);
				uinteger_t startLoc = access_connection_(connection->pDelayStart, delta_t, connection->nNum, nid);
				uinteger_t synapseNum = access_connection_(connection->pDelayNum, delta_t, connection->nNum, nid);
				// if (threadIdx.x == 0) {
				// 	gLayerInput[nid]++;
				// }
				// for (uinteger_t j=threadIdx.x; j<synapseNum; j += blockDim.x) {
				for (uinteger_t j = 0; j < synapseNum; ++j) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					uinteger_t sid = j+startLoc;
					real weight = data->pWeight[connection->pSidMap[sid]];
					atomicAdd(&(buffer[connection->dst[sid]]), weight);
				}
			}
		}
		__syncthreads();
	}
}

__global__ void update_static_2l(Connection *connection, StaticData *data, real *buffer, const uinteger_t *firedTable, const uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
{
	int delayLength = connection->maxDelay - connection->minDelay + 1;


	int d_para = gridDim.y;
	int d_offset = delayLength % d_para;
	// if (d_offset == 0) d_offset = d_para;
	// int d_offset = (delayLength % d_para + d_para) % (d_para + 1) + 1; 
	int d_avg_floor = (delayLength + d_para - 1) / d_para - 1;
	int d_start = 0;
	int d_end = 0;
	int d_idx = blockIdx.y;
	// printf("d_idx: %d\n", &d_idx);
	if (d_idx < d_offset) {
		d_start = d_idx * (d_avg_floor + 1);
		d_end = d_start + d_avg_floor + 1;
	// } else if (d_idx < delayLength) {
	} else if (d_idx < d_para) {
		d_start = d_idx * d_avg_floor + d_offset;
		d_end = d_start + d_avg_floor;
	} else {
		d_start = 0;
		d_end = 0;
	}

	for (int delta_t=d_start; delta_t<d_end; delta_t++) {
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
				nid_offset = time_idx * firedTableCap + block_idx * num_per_block;
			} else if (block_idx < gridDim.x) {
				fired_size_block = npb_minus_1;
				nid_offset = time_idx * firedTableCap + block_idx * npb_minus_1 + full_offset;
			} else {
				fired_size_block = 0;
				nid_offset = 0;
			}

			for (uinteger_t idx = 0; idx < fired_size_block; idx++) {
#ifdef DEBUG
				if (nid_offset + idx > firedSize + time_idx * firedTableCap) {
					printf("over flow %lld + %ld >= %lld, from (%d, %d)\n", (long long)nid_offset, (long long)idx, (long long)(firedSize + time_idx * firedTableCap), blockIdx.x, threadIdx.x);
					printf("%lld %d %d %ld %d %lld %ld\n", (long long)firedSize, gridDim.x, time_idx, firedTableCap, block_idx, (long long)npb_minus_1, (long long)full_offset);
				}
#endif
				uinteger_t nid = firedTable[nid_offset + idx];

				// uinteger_t startLoc = access_(connection->pDelayStart, delta_t, nid);
				// uinteger_t synapseNum = access_(connection->pDelayNum, delta_t, nid);
				uinteger_t startLoc = access_connection_(connection->pDelayStart, delta_t, connection->nNum, nid);
				uinteger_t synapseNum = access_connection_(connection->pDelayNum, delta_t, connection->nNum, nid);
				
				// uinteger_t synapseNum = connection->pDelayNum[delta_t * connection->nNum + nid];
				// if (threadIdx.x == 0) {
				// 	gLayerInput[nid]++;
				// }
				for (uinteger_t j=threadIdx.x; j<synapseNum; j += blockDim.x) {
					//int sid = connection->pSynapsesIdx[j+startLoc];
					uinteger_t sid = j+startLoc;
					real weight = data->pWeight[connection->pSidMap[sid]];
					atomicAdd(&(buffer[connection->dst[sid]]), weight);
				}
			}
		}
		__syncthreads();
	}
}

void cudaUpdateStatic(Connection * connection, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time, BlockSize *pSize)
{
	// std::cout << pSize->gridSize << " " << pSize->blockSize << std::endl;
	//update_static_hit<<<pSize->gridSize, pSize->blockSize>>>((StaticData*)data, num, start_id);
	//reset_active_synapse<<<1, 1>>>();
	// update_denser_static_hit<<<pSize->gridSize, pSize->blockSize>>>((Connection *)connection,  (StaticData *)data, buffer, firedTable, firedTableSizes, firedTableCap, num, start_id, time);
	// std::cout << "NUMBER in updateStatic: " << num << std::endl;


	update_dense_static_hit<<<pSize->gridSize, pSize->blockSize>>>((Connection *)connection,  (StaticData *)data, buffer, firedTable, firedTableSizes, firedTableCap, num, start_id, time);

	// uint3 block, grid;
	// grid.z = 1;
	// grid.y = 256;
	// grid.x = pSize->gridSize / grid.y;
	// block.z = 1;
	// block.y = 1;
	// block.x = pSize->blockSize;
	// update_static_2l<<<grid, block>>>((Connection *)connection,  (StaticData *)data, buffer, firedTable, firedTableSizes, firedTableCap, num, start_id, time);
}
