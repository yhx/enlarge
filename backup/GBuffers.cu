
#include "helper_gpu.h"
#include "runtime.h"

#include "GBuffers.h"

__global__ void init_buffers(/*int *c_gTimeTable, real *c_gNeuronInput, real *c_gNeuronInput_I, int *c_gFiredTable, int *c_gFiredTableSizes, */int *c_gActiveTable/*, int *c_gSynapsesActiveTable, int *c_gSynapsesLogTable*/) 
{
	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
		// gActiveTable = c_gActiveTable;
		// gActiveTableSize = 0;
		// gCurrentIdx = 0;
		// gCurrentCycle = 0;
		// gTimeTable = c_gTimeTable;
		// gNeuronInput = c_gNeuronInput;
		// gNeuronInput_I = c_gNeuronInput_I;
		// gFiredTable = c_gFiredTable;
		// gFiredTableSize = 0;
		// gFiredTableSizes = c_gFiredTableSizes;
		// gSynapsesActiveTable = c_gSynapsesActiveTable;
		// gSynapsesActiveTableSize = 0;
		// gSynapsesLogTable = c_gSynapsesLogTable;
	}
}

__global__ void init_log_buffers(uinteger_t *layer_input, real *x_input, int *fire_count)
{
	if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
		gLayerInput = layer_input;
		gXInput = x_input;
		gFireCount = fire_count;
	}
}


GBuffers* alloc_buffers(size_t neuron_num, size_t synapse_num, int maxDelay, real dt) 
{
	// int timeTableCap = deltaDelay+1;
	// checkCudaErrors(cudaMemcpyToSymbol(MAX_DELAY, &maxDelay, sizeof(int)));
	// checkCudaErrors(cudaMemcpyToSymbol(MIN_DELAY, &minDelay, sizeof(int)));
	// checkCudaErrors(cudaMemcpyToSymbol(gTimeTableCap, &timeTableCap, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(gFiredTableCap, &neuron_num, sizeof(int)));
	// checkCudaErrors(cudaMemcpyToSymbol(gSynapsesTableCap, &synapse_num, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(DT, &dt, sizeof(real)));

	GBuffers *ret = (GBuffers*)malloc(sizeof(GBuffers));
	memset(ret, 0, sizeof(GBuffers));

	ret->c_gNeuronInput = gpuMalloc<real>(neuron_num);
	ret->c_gNeuronInput_I = gpuMalloc<real>(neuron_num);

	ret->c_gFiredTable = gpuMalloc<uinteger_t>(neuron_num * (maxDelay+1));
	ret->c_gFiredTableSizes = gpuMalloc<uinteger_t>(maxDelay+1);

	ret->c_gLayerInput = gpuMalloc<uinteger_t>(neuron_num);
	ret->c_gXInput = gpuMalloc<real>(neuron_num);
	ret->c_gFireCount = gpuMalloc<int>(neuron_num);
	ret->c_neuronsFired = hostMalloc<uinteger_t>(neuron_num);


	// checkCudaErrors(cudaMallocHost((void**)(&ret->c_neuronsFired), sizeof(int)*(neuron_num)));

	// checkCudaErrors(cudaMalloc((void**)&(ret->c_gActiveTable), sizeof(int)*(neuron_num)));
	// checkCudaErrors(cudaMemset(ret->c_gActiveTable, 0, sizeof(int)*(neuron_num)));

	// checkCudaErrors(cudaMalloc((void**)&(ret->c_gSynapsesActiveTable), sizeof(int)*(synapse_num)));
	// checkCudaErrors(cudaMemset(ret->c_gSynapsesActiveTable, 0, sizeof(int)*(synapse_num)));

	// checkCudaErrors(cudaMalloc((void**)&(ret->c_gSynapsesLogTable), sizeof(int)*(synapse_num)));
	// checkCudaErrors(cudaMemset(ret->c_gSynapsesLogTable, 0, sizeof(int)*(synapse_num)));


	//checkCudaErrors(cudaMalloc((void**)&ret->c_gTimeTable, sizeof(int)*(deltaDelay+1)));
	//checkCudaErrors(cudaMemset(ret->c_gTimeTable, 0, sizeof(int)*(deltaDelay+1)));

	// checkCudaErrors(cudaMallocHost((void**)(&ret->c_neuronsFired), sizeof(int)*(neuron_num)));
	// checkCudaErrors(cudaMallocHost((void**)(&ret->c_synapsesFired), sizeof(int)*(synapse_num)));

	// init_buffers<<<1, 1, 0>>>(/*ret->c_gTimeTable,*/ ret->c_gNeuronInput, ret->c_gNeuronInput_I, ret->c_gFiredTable, ret->c_gFiredTableSizes, ret->c_gActiveTable/*, ret->c_gSynapsesActiveTable, ret->c_gSynapsesLogTable*/);
	// init_buffers<<<1, 1, 0>>>(ret->c_gActiveTable);

	init_log_buffers<<<1, 1, 0>>>(ret->c_gLayerInput, ret->c_gXInput, ret->c_gFireCount);

	return ret;
}

// void init_buffers(GBuffers * buf) {
// 	init_buffers<<<1, 1, 0>>>(/*buf->c_gTimeTable,*/ buf->c_gNeuronInput, buf->c_gNeuronInput_I, buf->c_gFiredTable, buf->c_gFiredTableSizes, buf->c_gActiveTable/*, buf->c_gSynapsesActiveTable, buf->c_gSynapsesLogTable*/);
// 
// 	init_log_buffers<<<1, 1, 0>>>(buf->c_gLayerInput, buf->c_gXInput, buf->c_gFireCount);
// }

int free_buffers(GBuffers *buf) 
{
	gpuFree(buf->c_gNeuronInput);
	gpuFree(buf->c_gNeuronInput_I);
	gpuFree(buf->c_gFiredTable);
	gpuFree(buf->c_gFiredTableSizes);
	hostFree(buf->c_neuronsFired);
	// checkCudaErrors(cudaFree(buf->c_gActiveTable));
	// checkCudaErrors(cudaFree(buf->c_gSynapsesActiveTable));
	// checkCudaErrors(cudaFree(buf->c_gSynapsesLogTable));
	// hostFree(buf->c_synapsesFired);

	return 0;
}
