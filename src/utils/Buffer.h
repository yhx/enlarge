
#ifndef BUFFER_H
#define BUFFER_H

#include "../base/constant.h"

class Buffer {
public:
	Buffer(size_t buffer_size, size_t n_num, int max_delay);
	~Buffer();


	size_t _fire_table_cap;
	// Input Current
	real *_data_buffer;
	// Neuron Tables
	uinteger_t *_fire_table;
	uinteger_t *_fired_sizes;

	uinteger_t *_neurons;

	size_t *_fire_count;
};

#endif // BUFFER_H
