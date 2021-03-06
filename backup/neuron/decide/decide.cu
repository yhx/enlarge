
#include "../../gpu_utils/runtime.h"

// #include "GDecideNeurons.h"
#include "decide.h"

__global__ void update_decide_neuron(GDecideNeurons *d_neurons, int num, int start_id, int time)
{

	int currentIdx = time % (MAX_DELAY+1);
	__shared__ int fire_table_t[MAXBLOCKSIZE];
	__shared__ volatile unsigned int fire_cnt;

	if (threadIdx.x == 0) {
		fire_cnt = 0;
	}
	__syncthreads();

	const int PERIOD = 10;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int idx = tid; idx < num; idx += blockDim.x * gridDim.x) {
		bool fired = false;
		int test_loc = 0;
		int gnid = idx + start_id;

		int max_idx = 0;
		real max = d_neurons->p_fire_rate[0];
		for (int i=1; i<num; i++) {
			if (d_neurons->p_fire_rate[i] > max) {
				max = d_neurons->p_fire_rate[i];
				max_idx = i;
			}
		}

		//fired = (PERIOD * d_neurons->p_fire_rate[idx]) > (d_neurons->p_fire_count[idx]);
		fired = (idx == max_idx);
		gFireCount[gnid] += fired;

		d_neurons->p_tmp_rate[idx] += gNeuronInput[gnid] + gNeuronInput_I[gnid];

		if (time % PERIOD == 0) {
			d_neurons->p_fire_rate[idx] = 0.5 *d_neurons->p_fire_rate[idx] + d_neurons->p_tmp_rate[idx] * 0.05;
			d_neurons->p_tmp_rate[idx] = 0;
			d_neurons->p_fire_count[idx] = 0;
		}

		for (int i=0; i<2; i++) {
			if (fired) {
				test_loc = atomicAdd((int*)&fire_cnt, 1);
				if (test_loc < MAXBLOCKSIZE) {
					fire_table_t[test_loc] = start_id + idx;
					d_neurons->p_fire_count[idx] = d_neurons->p_fire_count[idx] + 1;
					fired = false;
				}
			}
			__syncthreads();
			if (fire_cnt >= MAXBLOCKSIZE) {
				commit2globalTable(fire_table_t, MAXBLOCKSIZE, gFiredTable, &(gFiredTableSizes[currentIdx]), gFiredTableCap*currentIdx);
				if (threadIdx.x == 0) {
					fire_cnt = 0;
				}
			}
			__syncthreads();
		}
		gNeuronInput[gnid] = 0;
		gNeuronInput_I[gnid] = 0;
		__syncthreads();
	}
	__syncthreads();

	if (fire_cnt > 0) {
		commit2globalTable(fire_table_t, fire_cnt, gFiredTable, &(gFiredTableSizes[currentIdx]), gFiredTableCap*currentIdx);
		if (threadIdx.x == 0) {
			fire_cnt = 0;
		}
	}
}

int cudaUpdateDecide(void *data, int num, int start_id, int time, BlockSize *pSize)
{
	update_decide_neuron<<<pSize->gridSize, pSize->blockSize>>>((GDecideNeurons*)data, num, start_id, time);

	return 0;
}

