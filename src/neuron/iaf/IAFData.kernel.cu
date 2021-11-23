
#include "IAFData.h"

#include "../../../msg_utils/helper/helper_gpu.h"
#include "../../gpu_utils/runtime.h"
#include "../../net/Connection.h"


__global__ void update_all_iaf_neuron(Connection *connection, IAFData *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time)
{
	int currentIdx = time % (connection->maxDelay + 1);
	__shared__ uinteger_t fire_table_t[MAX_BLOCK_SIZE];
	__shared__ volatile uinteger_t fire_cnt;

	if (threadIdx.x == 0) {
		fire_cnt = 0;
	}

	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t idx = 0; idx < num; idx += blockDim.x * gridDim.x) {
		bool fired = false;
		uinteger_t testLoc = 0;

		size_t nid = idx + tid;
		size_t gnid = offset + nid; 
		if (nid < num) {
			if (data->pRefracStep[nid] <= 0) {  // neuron not refractory, so evolve V
				data->pV_m[nid] = (data->pV_m[nid] - data->pE_L[nid]) * data->pP22[nid] 
								+ data->pi_syn_ex[nid] * data->pP21ex[nid]  // input excitatory current
								+ data->pi_syn_in[nid] * data->pP21in[nid]  // input inhibitory current
								+ ( data->pI_e[nid] + data->pi_0[nid] ) * data->pP20[nid] + data->pE_L[nid];
			} else {  // neuron is absolute refractory
				data->pRefracStep[nid] = data->pRefracStep[nid] - 1;
			}

			// exponential decaying PSCs
			data->pi_syn_ex[nid] *= data->pP11ex[nid];
			data->pi_syn_in[nid] *= data->pP11in[nid];

			// add evolution of presynaptic input current
			data->pi_syn_ex[nid] += (1.0 - data->pP11ex[nid]) * data->pi_1[nid];

			// the spikes arriving at T+1 have an immediate effect on the state of the neuron
			real weighted_spikes_ex = buffer[gnid];
			real weighted_spikes_in = buffer[gnid + num];

			data->pi_syn_ex[nid] += weighted_spikes_ex;
			data->pi_syn_in[nid] += weighted_spikes_in; 

			fired = data->pV_m[nid] >= data->pTheta[nid];
			data->_fire_count[gnid] += fired;

			if (fired) {  // update fire table if fired
				testLoc = atomicAdd((uinteger_t *)&fire_cnt, 1);
				if (testLoc < MAX_BLOCK_SIZE) {
					fire_table_t[testLoc] = gnid;
					fired = false;
				}
				// firedTable[firedTableSizes[currentIdx] + firedTableCap * currentIdx] = gnid;
				// firedTableSizes[currentIdx]++;
	
				data->pRefracStep[nid] = data->pRefracTime[nid];
				data->pV_m[nid] = data->pV_reset[nid];
			} 
	
			data->pi_0[nid] = 0;
			data->pi_1[nid] = 0;
			
			// clear buffer
			buffer[gnid] = 0;		
			buffer[num + gnid] = 0;
		}
		
		// 如果当前fire_cnt的数量大于等于最大的block大小，那么直接commit一次，并将fire_cnt置为0
		__syncthreads();
		if (fire_cnt >= MAX_BLOCK_SIZE) {
			commit2globalTable(fire_table_t, static_cast<uinteger_t>(MAX_BLOCK_SIZE), firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
			if (threadIdx.x == 0) {
				fire_cnt = 0;
			}
		}

		__syncthreads();
		if (fired) {
			testLoc = atomicAdd((uinteger_t*)&fire_cnt, 1);
			if (testLoc < MAX_BLOCK_SIZE) {
				fire_table_t[testLoc] = gnid;
				fired = false;
			}
		}
		__syncthreads();
		if (fire_cnt >= MAX_BLOCK_SIZE) {
			commit2globalTable(fire_table_t, static_cast<uinteger_t>(MAX_BLOCK_SIZE), firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
			if (threadIdx.x == 0) {
				fire_cnt = 0;
			}
		}
		__syncthreads();

	}

	// 如果当前fire_cnt大于0，则更新firedTable相关的信息
	if (fire_cnt > 0) {
		commit2globalTable(fire_table_t, fire_cnt, firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
		if (threadIdx.x == 0) {
			fire_cnt = 0;
		}
	}
	__syncthreads();
}

void cudaUpdateIAF(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time, BlockSize *pSize)
{
	update_all_iaf_neuron<<<pSize->gridSize, pSize->blockSize>>>(conn, (IAFData*)data, buffer, firedTable, firedTableSizes, firedTableCap, num, offset, time);
}
