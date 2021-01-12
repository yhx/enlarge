
#include <assert.h>

#include "../third_party/cuda/helper_cuda.h"
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

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_offset), sizeof(int)*num_p_1));
	checkCudaErrors(cudaMemcpy(gpu->_recv_offset, data->_recv_offset, sizeof(int)*num_p_1, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_start), sizeof(int)*(size+num)));
	checkCudaErrors(cudaMemcpy(gpu->_recv_start, data->_recv_start, sizeof(int)*(size+num), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_num), sizeof(int)*num));
	checkCudaErrors(cudaMemset(gpu->_recv_num, 0, sizeof(int)*num));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_offset), sizeof(int)*num_p_1));
	checkCudaErrors(cudaMemcpy(gpu->_send_offset, data->_send_offset, sizeof(int)*num_p_1, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_start), sizeof(int)*(size+num)));
	checkCudaErrors(cudaMemcpy(gpu->_send_start, data->_send_start, sizeof(int)*(size+num), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_num), sizeof(int)*num));
	checkCudaErrors(cudaMemset(gpu->_send_num, 0, sizeof(int)*num));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_data), sizeof(int)*(data->_recv_offset[num])));
	checkCudaErrors(cudaMemset(gpu->_recv_data, 0, sizeof(int)*(data->_recv_offset[num])));


	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_data), sizeof(int)*(data->_send_offset[num])));
	checkCudaErrors(cudaMemset(gpu->_send_data, 0, sizeof(int)*(data->_send_offset[num])));

	return gpu;
}


int freeCNDGPU(CrossNodeData *data) 
{
	cudaFree(data->_recv_offset);
	cudaFree(data->_recv_start);
	cudaFree(data->_recv_num);
	cudaFree(data->_recv_data);

	cudaFree(data->_send_offset);
	cudaFree(data->_send_start);
	cudaFree(data->_send_num);
	cudaFree(data->_send_data);

	data->_node_num = 0;
	data->_min_delay = 0;
	free(data);
	data = NULL;
	return 0;
}

__global__ void cuda_gen_cnd(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, int *send_data, int *send_offset, int *send_start, int node_num, int time, int min_delay, int delay)
{
	__shared__ int cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile int cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int delayIdx = time % (conn->maxDelay-conn->minDelay+1);
	int delayIdx = time % (conn->maxDelay+1);
	int fired_size = firedTableSizes[delayIdx];
	for (int node = 0; node < node_num; node++) {
		for (int idx = tid; idx < fired_size; idx += blockDim.x * gridDim.x) {
			int nid = firedTable[gFiredTableCap*delayIdx + idx];
			int tmp = idx2index[nid];
			
			if (tmp >= 0) {
				int map_nid = crossnode_index2idx[tmp*node_num + node];
				if (map_nid >= 0) {
					int test_loc = atomicAdd((int*)&cross_cnt, 1);
					if (test_loc < MAX_BLOCK_SIZE) {
						cross_neuron_id[test_loc] = map_nid;
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				int idx_t = node * (min_delay+1) + delay + 1;
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

__global__ void update_cnd_start(int *start, int node, int min_delay, int curr_delay) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=tid; i<node; i++) {
		start[i*(min_delay+1)+curr_delay+2] = start[i*(min_delay+1)+curr_delay+1];
	}
}


void cudaGenerateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeData *cnd, int node_num, int time, int delay, int gridSize, int blockSize) 
{
	cuda_gen_cnd<<<gridSize, blockSize>>>(conn, firedTable, firedTableSizes, idx2index, crossnode_index2idx, cnd->_send_data, cnd->_send_offset, cnd->_send_start, node_num, time, cnd->_min_delay, delay);
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
	cudaMemset(gpu->_recv_start, 0, sizeof(int)*(size+node_num));
	cudaMemset(gpu->_send_start, 0, sizeof(int) * (size+node_num));

	memset(cpu->_recv_num, 0, sizeof(int) * node_num);
	memset(cpu->_send_num, 0, sizeof(int) * node_num);
	return 0;
}
