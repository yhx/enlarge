
#include "../utils/helper_c.h"
#include "Buffer.h"

// size_t *c_gFiredCount = NULL;

Buffer::Buffer()
{
	_gpu = -1;
	_delay = 0;
	_data_size = 0;
	_fire_table_cap = 0;

	_fire_table = NULL;
	_fired_sizes = NULL;
	// _neurons = NULL;
	// _fire_count = NULL;
	_data = NULL;

	_gpu_array = NULL;
}

Buffer::Buffer(size_t data_size, size_t n_num, int max_delay)
{
	_gpu = -1;
	_delay = max_delay;
	_data_size = data_size;
	_fire_table_cap = n_num;

	_fire_table =  new uinteger_t[n_num * (max_delay+1)]();
	_fired_sizes = new uinteger_t[max_delay + 1]();
	// _neurons = new uinteger_t[n_num]();
	// _fire_count = new size_t[n_num]();

	_data= new real[data_size]();
	
	// c_gFiredCount = _fire_count;

	_gpu_array = NULL;
}

Buffer::~Buffer()
{
	_delay = 0;
	if (_fire_table_cap > 0) {
		delete [] _fire_table;
		delete [] _fired_sizes;
		// delete [] _neurons;
		// delete [] _fire_count;
	}
	_fire_table_cap = 0;

	if (_data_size > 0) {
		delete [] _data;
	}
	_data_size = 0;
	
#ifdef USE_GPU
	if (_gpu_array) {
		free_gpu();
	}
#else
	assert(!_gpu_array);
#endif
}

#ifndef USE_GPU
Buffer::Buffer(size_t data_size, size_t n_num, int max_delay): Buffer(data_size, n_num, max_delay, int gpu)
{
}

void Buffer::free_gpu()
{
}
#endif

