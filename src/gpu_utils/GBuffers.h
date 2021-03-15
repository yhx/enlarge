
#include "../utils/constant.h"

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

	// int *c_neuronsFired;
	// int *c_synapsesFired;

	uinteger_t *c_gLayerInput;
	real *c_gXInput;

	int *c_gFireCount;
};

// void init_buffers(GBuffers * buf);
GBuffers* alloc_buffers(int neuron_num, int synapse_num, int maxDelay, real dt);
int free_buffers(GBuffers *buf);
