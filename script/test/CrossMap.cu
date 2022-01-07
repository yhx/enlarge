#include <assert.h>
#include "../gpu_utils/helper_gpu.h"
#include "CrossMap.h"


void CrossMap::free_gpu()
{
	if (_gpu_array && num > 0) {
		gpuFree(_gpu_array->data);
	}
	delete _gpu_array;
	_gpu_array = NULL;
}

int to_gpu() 
{
	if (!_gpu_array) {
		_gpu_array = new CrossMap();
		_gpu_array->_is_view = _is_view;
		_gpu_array->_num = _num;
		_gpu_array->_crossSize = copyToGPU(_data, num);
		_gpu_array->_num = copyToGPU(_data, num);


		_gpu_array->_crossnodeIndex2idx = copyToGPU(_data, num);
		_gpu_array->_idx2index = copyToGPU(_data, num);


	} else {
		assert(_gpu_array->_num == _num);
		copyToGPU(_gpu_array->_crossSize, _crossSize, num);
		copyToGPU(_gpu_array->_num, _num, num);


		copyToGPU(_gpu_array->_crossnodeIndex2idx, _crossnodeIndex2idx, num);
		copyToGPU(_gpu_array->_idx2index, _idx2index, num);


	}
	return 0;
}


int from_gpu()
{
	if (!_gpu_array) {
		printf("No Data on GPU!\n");
		return -1;
	}
	copyFromGPU(_crossSize, _gpu_array->_crossSize, num);
	copyFromGPU(_num, _gpu_array->_num, num);


	copyFromGPU(_crossnodeIndex2idx, _gpu_array->_crossnodeIndex2idx, num);
	copyFromGPU(_idx2index, _gpu_array->_idx2index, num);


	return 0;
}
