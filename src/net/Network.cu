
#include "../utils/helper_c.h"
#include "../gpu_utils/helper_gpu.h"
#include "Network.h"

//Alloc cross device gmem, will set device to 0 before return
CrossThreadDataGPU* Network::arrangeCrossGPUData()
{
	CrossThreadDataGPU * cross_data = malloc_c<CrossThreadDataGPU>();
	
	unsigned node_sq = _node_num * _node_num;
	cross_data->_maxNum = malloc_c<uinteger_t>(node_sq);
	cross_data->_firedArrays = malloc_c<uinteger_t*>(node_sq);
	cross_data->_firedNum = hostMalloc<uinteger_t>(node_sq);

	for (unsigned int i=0; i<_node_num; i++) {
		for (unsigned int j=0; j<_node_num; j++) {
			// i->j 
			checkCudaErrors(cudaSetDevice(j));
			unsigned int i2j = i * _node_num + j;
			cross_data->_firedNum[i2j] = 0;

			size_t count = 0;
			for (auto iter = _crossnodeNeuronsSend[i].begin(); iter != _crossnodeNeuronsSend[i].end(); iter++) {
				if (_crossnodeNeuronsRecv[j].find(*iter) != _crossnodeNeuronsRecv[j].end()) {
					count++;
				}
			}
			cross_data->_maxNum[i2j] = count;
			if (count > 0) {
				cross_data->_firedArrays[i2j] = gpuMalloc<uinteger_t>(count);
			} else {
				cross_data->_firedArrays[i2j] = NULL;
			}
		}
	}

	checkCudaErrors(cudaSetDevice(0));

	return cross_data;
}

