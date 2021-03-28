
#ifndef BUFFER_H
#define BUFFER_H

#include "../base/constant.h"

class Buffer {
public:
	Buffer();
	Buffer(size_t data_size, size_t n_num, int max_delay);
	Buffer(size_t data_size, size_t n_num, int max_delay, int gpu);
	~Buffer();

	void free_gpu();

	int _gpu;
	int _delay;
	size_t _data_size;
	size_t _fire_table_cap;
	// Neuron Tables
	uinteger_t *_fire_table;
	uinteger_t *_fired_sizes;

	uinteger_t *_neurons;

	// Input Current
	real *_data;

	// size_t *_fire_count;

	Buffer *_gpu_array;
};

#endif // BUFFER_H
