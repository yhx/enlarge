

#ifdef USE_GPU
#include "../../msg_utils/helper/helper_gpu.h"
#endif

#include "Data.h"

#ifdef USE_GPU
int Data::to_gpu() 
{
	if (!_gpu) {
		// _gpu = new Data();
		_gpu->_is_view = _is_view;
		_gpu->_num = _num;

		// _gpu->_data = copyToGPU(_data, _num);
	} else {
		_gpu->_is_view = _is_view;
		_gpu->_num = _num;

		// copyToGPU(_gpu->_data, _data, _num);
	}
	return 0;
}

int Data::from_gpu() 
{
	if (!_gpu) {
		printf("No Data on GPU!\n");
		return -1;
	}

	// copyFromGPU(_data, _gpu->data, _num);
	return 0;
}
#endif
