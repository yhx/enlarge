
#include <assert.h>

#include "../third_party/cuda/helper_cuda.h"

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

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_data), sizeof(int)*(data->_recv_offset[num])));
	checkCudaErrors(cudaMemset(gpu->_recv_data, 0, sizeof(int)*(data->_recv_offset[num])));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_offset), sizeof(int)*num_p_1));
	checkCudaErrors(cudaMemcpy(gpu->_send_offset, data->_send_offset, sizeof(int)*num_p_1, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_start), sizeof(int)*(size+num)));
	checkCudaErrors(cudaMemcpy(gpu->_send_start, data->_send_start, sizeof(int)*(size+num), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_num), sizeof(int)*num));
	checkCudaErrors(cudaMemset(gpu->_send_num, 0, sizeof(int)*num));

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
