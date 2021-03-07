
#include "../third_party/cuda/helper_cuda.h"
#include "Network.h"

//Alloc cross device gmem, will set device to 0 before return
CrossThreadDataGPU* Network::arrangeCrossThreadDataGPU(int node_num)
{
	CrossThreadDataGPU * cross_data = static_cast<CrossThreadDataGPU*>(malloc(sizeof(CrossThreadDataGPU)));
	assert(cross_data != NULL);

	cross_data->_maxNum = static_cast<int*>(malloc(sizeof(int)*node_num*node_num));
	assert(cross_data->_maxNum != NULL);
	cross_data->_firedArrays = static_cast<int**>(malloc(sizeof(int*)*node_num*node_num));
	assert(cross_data->_firedArrays != NULL);

	checkCudaErrors(cudaMallocHost((void**)&(cross_data->_firedNum), sizeof(int)*node_num*node_num));
	checkCudaErrors(cudaMemset(cross_data->_firedNum, 0, sizeof(int)*node_num*node_num));


	for (int i=0; i<_node_num; i++) {
		for (int j=0; j<_node_num; j++) {
			// i->j 
			checkCudaErrors(cudaSetDevice(j));
			int i2j = i * _node_num + j;
			cross_data->_firedNum[i2j] = 0;

			int count = 0;
			for (auto iter = _crossnodeNeuronsSend[i].begin(); iter != _crossnodeNeuronsSend[i].end(); iter++) {
				if (_crossnodeNeuronsRecv[j].find(*iter) != _crossnodeNeuronsRecv[j].end()) {
					count++;
				}
			}
			cross_data->_maxNum[i2j] = count;
			if (count > 0) {
				checkCudaErrors(cudaMalloc((void**)&(cross_data->_firedArrays[i2j]), sizeof(int)*count));
			} else {
				cross_data->_firedArrays[i2j] = NULL;
			}
		}
	}

	checkCudaErrors(cudaSetDevice(0));

	return cross_data;
}

