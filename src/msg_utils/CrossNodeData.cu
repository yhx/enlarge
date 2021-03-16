
#include <assert.h>

#include "../gpu_utils/helper_gpu.h"
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

    gpu->_recv_offset = copyToGPU<uinteger_t>(data->_recv_offset, num_p_1);
    gpu->_recv_start = copyToGPU<uinteger_t>(data->_recv_start, size+num);
    gpu->_recv_num = gpuMalloc<uinteger_t>(num);

    gpu->_recv_data = gpuMalloc<uinteger_t>(data->_recv_offset[num]);

    gpu->_send_offset = copyToGPU<uinteger_t>(data->_send_offset, num_p_1);
    gpu->_send_start = copyToGPU<uinteger_t>(data->_send_start, size+num);
    gpu->_send_num = gpuMalloc<uinteger_t>(num);

    gpu->_send_data = gpuMalloc<uinteger_t>(data->_send_offset[num]);

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

template<typename T1, typename T2>
__global__ void cuda_gen_cnd(T1 *idx2index, T1 *crossnode_index2idx, T2 *send_data, T1 *send_offset, T1 *send_start, T2 *firedTable, T2 *firedTableSizes, size_t firedTableCap, int max_delay, int min_delay, int node_num, int time)
{
	__shared__ T1 cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile unsigned int cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int delayIdx = time % (conn->maxDelay-conn->minDelay+1);
	int delayIdx = time % (max_delay+1);
	int curr_delay = time % min_delay;
	T2 fired_size = firedTableSizes[delayIdx];
	for (int node = 0; node < node_num; node++) {
		for (size_t idx = tid; idx < fired_size; idx += blockDim.x * gridDim.x) {
			size_t nid = firedTable[gFiredTableCap*delayIdx + idx];
			T1 tmp = idx2index[nid];
			
			if (tmp >= 0) {
				T1 map_nid = crossnode_index2idx[tmp*node_num + node];
				if (map_nid >= 0) {
					unsigned int test_loc = atomicAdd((unsigned int*)&cross_cnt, 1);
					if (test_loc < MAX_BLOCK_SIZE) {
						cross_neuron_id[test_loc] = map_nid;
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				size_t idx_t = node * (min_delay+1) + curr_delay + 1;
				commit2globalTable(cross_neuron_id, cross_cnt, send_data, &(send_start[idx_t]), send_offset[node]);
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
	for (int i=tid; i<node; i++) {
		start[i*(min_delay+1)+curr_delay+2] = start[i*(min_delay+1)+curr_delay+1];
	}
}


template<typename T1, typename T2>
void cudaGenerateCND(T1 *idx2index, T1 *crossnode_index2idx, CrossNodeData *cnd, T2 *firedTable, T2 *firedTableSizes, size_t firedTableCap, int max_delay, int min_delay, int node_num, int time, int gridSize, int blockSize) 
{
	cuda_gen_cnd<<<gridSize, blockSize>>>(idx2index, crossnode_index2idx, cnd->_send_data, cnd->_send_offset, cnd->_send_start, firedTable, firedTableSizes, firedTableCap, max_delay, min_delay, node_num, time);
}

int fetch_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu)
{
	assert(cpu->_node_num == gpu->_node_num);
	assert(cpu->_min_delay == gpu->_min_delay);

	int num = cpu->_node_num;
	int size = cpu->_node_num * cpu->_min_delay;

	checkCudaErrors(cudaMemcpy(cpu->_send_start, gpu->_send_start, sizeof(int)*(size+num), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cpu->_send_data, gpu->_send_data, sizeof(int)*(cpu->_send_offset[num]), cudaMemcpyDeviceToHost));

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
	cudaMemset(gpu->_recv_start, 0, sizeof(uinteger_t)*(size+node_num));
	cudaMemset(gpu->_send_start, 0, sizeof(uinteger_t) * (size+node_num));

	memset(cpu->_recv_num, 0, sizeof(uinteger_t) * node_num);
	memset(cpu->_send_num, 0, sizeof(uinteger_t) * node_num);
	return 0;
}
