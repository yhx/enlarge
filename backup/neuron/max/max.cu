
#include "../../gpu_utils/runtime.h"

#include "max.h"


__global__ void update_max_neuron(GMaxNeurons *d_neurons, int num, int start_id, int time)
{
	int currentIdx = time % (MAX_DELAY+1);
	__shared__ int fire_table_t[MAXBLOCKSIZE];
	__shared__ volatile unsigned int fire_cnt;

	if (threadIdx.x == 0) {
		fire_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int idx = tid; idx < num; idx += blockDim.x * gridDim.x) {
		bool fired = false;
		int test_loc = 0;

		int input = (int)gNeuronInput[start_id + idx];
		gNeuronInput[start_id + idx] = 0;
		int test = 1;
		int record_offset = idx*d_neurons->max_N; 
		for (int i=0; i<d_neurons->p_N[idx]; i++) {
			if (input & test) {
				d_neurons->p_record[record_offset + i]++;
				if (d_neurons->p_record[record_offset + i] > d_neurons->p_count[idx]) {

					fired = true;
				}
			}
			test = test << 1;
		}

		gFireCount[start_id + idx] += fired;

		for (int i=0; i<2; i++) {
			if (fired) {
				test_loc = atomicAdd((int*)&fire_cnt, 1);
				if (test_loc < MAXBLOCKSIZE) {
					fire_table_t[test_loc] = start_id + idx;
					d_neurons->p_count[idx] = d_neurons->p_count[idx] + 1;
					fired = false;
				}
			}
			__syncthreads();
			if (fire_cnt >= MAXBLOCKSIZE) {
				commit2globalTable(fire_table_t, MAXBLOCKSIZE, gFiredTable, &(gFiredTableSizes[currentIdx]), gFiredTableCap*currentIdx);
				//advance_array_neuron(d_neurons, fire_table_t, MAXBLOCKSIZE, start_id);
				if (threadIdx.x == 0) {
					fire_cnt = 0;
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	if (fire_cnt > 0) {
		commit2globalTable(fire_table_t, fire_cnt, gFiredTable, &(gFiredTableSizes[currentIdx]), gFiredTableCap*currentIdx);
		//advance_array_neuron(d_neurons, fire_table_t, fire_cnt, start_id);
		if (threadIdx.x == 0) {
			fire_cnt = 0;
		}
	}

}

int cudaUpdateMax(void *data, int num, int start_id, int time, BlockSize *pSize)
{
	update_max_neuron<<<pSize->gridSize, pSize->blockSize>>>((GMaxNeurons*)data, num, start_id, time);

	return 0;
}

