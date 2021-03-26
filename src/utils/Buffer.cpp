
#include "../utils/helper_c.h"
#include "Buffer.h"

size_t *c_gFiredCount = NULL;


Buffer::Buffer(size_t buffer_size, size_t n_num, int max_delay)
{
	_fire_table_cap = n_num;
	_data_buffer = new real[buffer_size]();
	_fire_table =  new uinteger_t[n_num * (max_delay+1)]();
	_fired_sizes = new uinteger_t[max_delay + 1]();

	_neurons = new uinteger_t[n_num]();

	_fire_count = new size_t[n_num]();
	c_gFiredCount = _fire_count;
}

Buffer::~Buffer()
{
	_fire_table_cap = 0;
	delete [] _data_buffer;
	delete [] _fire_table;
	delete [] _fired_sizes;

	delete [] _neurons;

	delete [] _fire_count;
}
