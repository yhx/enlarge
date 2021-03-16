
#include "../base/constant.h"

struct GBuffers {
	// Neuron Arrays
	real *c_gNeuronInput;
	real *c_gNeuronInput_I;
	// Neuron Tables
	uinteger_t *c_gFiredTable;
	uinteger_t *c_gFiredTableSizes;
	// uinteger_t *c_gActiveTable;
	// Synapse Tables
	// int *c_gSynapsesActiveTable;
	// int *c_gSynapsesLogTable;

	uinteger_t *c_neuronsFired;
	// int *c_synapsesFired;

	uinteger_t *c_gLayerInput;
	real *c_gXInput;

	int *c_gFireCount;
};

// void init_buffers(GBuffers * buf);
GBuffers* alloc_buffers(size_t neuron_num, size_t synapse_num, int maxDelay, real dt);
int free_buffers(GBuffers *buf);
