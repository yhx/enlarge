
#include <assert.h>

#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "../gpu_utils/runtime.h"
#include "CrossNodeData.h"

CrossNodeData * copyCNDtoGPU(CrossNodeData *data)
{
	// TODO correct size;
	CrossNodeData *gpu = (CrossNodeData *)malloc(sizeof(CrossNodeData));
	assert(gpu != NULL);

	gpu->_node_num = data->_node_num;
	gpu->_min_delay = data->_min_delay;

	int num = data->_node_num;
	int size = data->_node_num * data->_min_delay;
	int num_p_1 = data->_node_num + 1;

    gpu->_recv_offset = TOGPU(data->_recv_offset, num_p_1);
    gpu->_recv_start = TOGPU(data->_recv_start, size+num);
    gpu->_recv_num = gpuMalloc<integer_t>(num);

    gpu->_recv_data = gpuMalloc<uinteger_t>(data->_recv_offset[num]);

    gpu->_send_offset = TOGPU(data->_send_offset, num_p_1);
    gpu->_send_start = TOGPU(data->_send_start, size+num);
    gpu->_send_num = gpuMalloc<integer_t>(num);

    gpu->_send_data = gpuMalloc<uinteger_t>(data->_send_offset[num]);

#ifdef PROF
	gpu->_cpu_wait_gpu = 0;
	gpu->_gpu_wait = 0;
	gpu->_gpu_time = 0;
	gpu->_comm_time = 0;
	gpu->_cpu_time = 0;
#endif

	return gpu;
}


int freeCNDGPU(CrossNodeData *data) 
{
	gpuFree(data->_recv_offset);
	gpuFree(data->_recv_start);
	gpuFree(data->_recv_num);
	gpuFree(data->_recv_data);

	gpuFree(data->_send_offset);
	gpuFree(data->_send_start);
	gpuFree(data->_send_num);
	gpuFree(data->_send_data);

	data->_node_num = 0;
	data->_min_delay = 0;
	free(data);
	data = NULL;
	return 0;
}

__global__ void cuda_gen_cnd(integer_t *idx2index, integer_t *crossnode_index2idx, uinteger_t *send_data, integer_t *send_offset, integer_t *send_start, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, int max_delay, int min_delay, int node_num, int time)
{
	__shared__ uinteger_t cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile integer_t cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int delayIdx = time % (conn->maxDelay-conn->minDelay+1);
	int delayIdx = time % (max_delay+1);
	int curr_delay = time % min_delay;
	uinteger_t fired_size = firedTableSizes[delayIdx];
	for (int node = 0; node < node_num; node++) {
		for (size_t idx = 0; idx < fired_size; idx += blockDim.x * gridDim.x) {
			if (idx + tid < fired_size) {
				size_t nid = firedTable[firedTableCap*delayIdx + idx + tid];
				integer_t tmp = idx2index[nid];

				if (tmp >= 0) {
					integer_t map_nid = crossnode_index2idx[tmp*node_num + node];
					if (map_nid >= 0) {
						unsigned int test_loc = atomicAdd((unsigned int*)&cross_cnt, 1);
						if (test_loc < MAX_BLOCK_SIZE) {
							cross_neuron_id[test_loc] = map_nid;
						}
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				size_t idx_t = node * (min_delay+1) + curr_delay + 1;
				commit2globalTable(cross_neuron_id, static_cast<integer_t>(cross_cnt), send_data, &(send_start[idx_t]), send_offset[node]);
				if (threadIdx.x == 0) {
					cross_cnt = 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

template<typename T>
__global__ void update_cnd_start(T *start, int node, int min_delay, int curr_delay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// for (int i=tid; i<node; i++) {
	if (tid < node) {
		start[tid*(min_delay+1)+curr_delay+2] = start[tid*(min_delay+1)+curr_delay+1];
	}
	// }
}


void cudaGenerateCND(integer_t *idx2index, integer_t *crossnode_index2idx, CrossNodeData *cnd, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, int max_delay, int min_delay, int node_num, int time, int gridSize, int blockSize) 
{
	cuda_gen_cnd<<<gridSize, blockSize>>>(idx2index, crossnode_index2idx, cnd->_send_data, cnd->_send_offset, cnd->_send_start, firedTable, firedTableSizes, firedTableCap, max_delay, min_delay, node_num, time);
}

int fetch_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu)
{
	assert(cpu->_node_num == gpu->_node_num);
	assert(cpu->_min_delay == gpu->_min_delay);

	int num = cpu->_node_num;
	int size = cpu->_node_num * cpu->_min_delay;

#ifdef PROF
	double ts = MPI_Wtime();
#endif
	checkCudaErrors(cudaMemcpy(cpu->_send_start, gpu->_send_start, sizeof(int)*(size+num), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cpu->_send_data, gpu->_send_data, sizeof(int)*(cpu->_send_offset[num]), cudaMemcpyDeviceToHost));
#ifdef PROF
	double te = MPI_Wtime();
	gpu->_cpu_wait_gpu += te - ts;
#endif

	return 0;
}

int update_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu, int curr_delay, MPI_Request *request) 
{
	int min_delay = cpu->_min_delay;
	if (curr_delay >= min_delay - 1) {
		fetch_cnd_gpu(gpu, cpu);
		msg_cnd(cpu, request);
	} else {
		update_cnd_start<<<(cpu->_node_num-1+MAX_BLOCK_SIZE)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(gpu->_send_start, cpu->_node_num, min_delay, curr_delay);
	}
	return 0;
}

int reset_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu)
{
	int node_num = cpu->_node_num;
	int size = cpu->_min_delay * cpu->_node_num;
	gpuMemset(gpu->_recv_start, 0, size+node_num);
	gpuMemset(gpu->_send_start, 0, size+node_num);

	memset_c(cpu->_recv_num, 0, node_num);
	memset_c(cpu->_send_num, 0, node_num);
	return 0;
}
