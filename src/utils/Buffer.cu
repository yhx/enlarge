
#include "../../msg_utils/helper/helper_gpu.h"
#include "../../msg_utils/msg_utils/GPUManager.h"
#include "Buffer.h"

Buffer::Buffer(size_t data_size, size_t n_num, int max_delay, int gpu): Buffer(data_size, n_num, max_delay)
{
	_gpu_array = new Buffer();

	_gpu_array->_gpu = gpu;
	_gpu_array->_delay = max_delay;
	_gpu_array->_data_size = data_size;
	_gpu_array->_fire_table_cap = n_num;

	// gpuSetDevice(_gpu_array->_gpu);
	gm.set(_gpu_array->_gpu);

	_gpu_array->_fire_table =  gpuMalloc<uinteger_t>(n_num * (max_delay+1));
	_gpu_array->_fired_sizes = gpuMalloc<uinteger_t>(max_delay + 1);
	// _gpu_array->_neurons = gpuMalloc<uinteger_t>(n_num);
	// _gpu_array->_fire_count = gpuMalloc<size_t>(n_num);

	_gpu_array->_data= gpuMalloc<real>(data_size);
}

void Buffer::free_gpu() {
	gpuSetDevice(_gpu_array->_gpu);
	if (_gpu_array->_fire_table_cap > 0) {
		gpuFree(_gpu_array->_fire_table);
		gpuFree(_gpu_array->_fired_sizes);
	//	gpuFree(_gpu_array->_neurons);
	//	gpuFree(_gpu_array->_fire_count);
	}
	_gpu_array->_fire_table_cap = 0;

	if (_gpu_array->_data_size > 0) {
		gpuFree(_gpu_array->_data);
	}
	_gpu_array->_data_size = 0;

	delete _gpu_array;
	_gpu_array = NULL;
}

