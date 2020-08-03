/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <mpi.h>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "../gpu_utils/mem_op.h"
// #include "../gpu_utils/gpu_utils.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/GBuffers.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "MultiNodeSimulator.h"
// #include "../gpu_utils/gpu_func.h"

using std::cout;
using std::endl;

MultiNodeSimulator::MultiNodeSimulator(Network *network, real dt) : Simulator(network, dt)
{
}

MultiNodeSimulator::~MultiNodeSimulator()
{
}

int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd) {
	char log_filename[512];
	sprintf(log_filename, "sim_%d.gpu.log", network->_nodeIdx); 
	FILE *log_file = fopen(log_filename, "w+");
	assert(log_file != NULL);

	char v_filename[512];
	sprintf(v_filename, "v_%d.gpu.data", network->_nodeIdx); 
	FILE *v_file = fopen(v_filename, "w+");
	assert(v_file != NULL);

	checkCudaErrors(cudaSetDevice(network->_nodeIdx));

	GNetwork *pNetCPU = network->_network;
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);

	CrossNodeData * cnd_gpu = copyCNDtoGPU(cnd);

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->pConnection->nNum;
	int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d NeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, nodeSynapseNum);

	int maxDelay = pNetCPU->pConnection->maxDelay;
	int minDelay = pNetCPU->pConnection->minDelay;

	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);


	GBuffers *buffers = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->pConnection->maxDelay, network->_dt);

	BlockSize *updateSize = getBlockSize(allNeuronNum, nodeSynapseNum);

#ifdef LOG_DATA
	real *c_vm = hostMalloc<real>(nodeNeuronNum);
	int life_idx = getIndex(c_pNetGPU->pNTypes, nTypeNum, LIF);
	int copy_idx = -1;
	real *c_g_vm = NULL;

	if (life_idx >= 0) {
		LIFData *c_g_lif = copyFromGPU<LIFData>(static_cast<LIFData *>(c_pNetGPU->ppNeurons[life_idx]), 1);
		c_g_vm = c_g_lif->pV_m;
		copy_idx = life_idx;
	} else {
	}
#endif

	for (int i=0; i<nTypeNum; i++) {
		cout << "Thread " << network->_nodeIdx << " " << c_pNetGPU->pNTypes[i] << ": <<<" << updateSize[c_pNetGPU->pNTypes[i]].gridSize << ", " << updateSize[c_pNetGPU->pNTypes[i]].blockSize << ">>>" << endl;
	}
	for (int i=0; i<sTypeNum; i++) {
		cout << "Thread " << network->_nodeIdx << " " << c_pNetGPU->pSTypes[i] << ": <<<" << updateSize[c_pNetGPU->pSTypes[i]].gridSize << ", " << updateSize[c_pNetGPU->pSTypes[i]].blockSize << ">>>" << endl;
	}

	int * c_g_idx2index = copyToGPU<int>(network->_crossnodeMap->_idx2index, allNeuronNum);
	int * c_g_cross_index2idx = copyToGPU<int>(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossSize);
	int * c_g_global_cross_data = gpuMalloc<int>(allNeuronNum * network->_nodeNum);
	int * c_g_fired_n_num = gpuMalloc<int>(network->_nodeNum);

	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);
	for (int time=0; time<network->_simCycle; time++) {
		update_time<<<1, 1>>>(c_pNetGPU->pConnection, time, buffers->c_gFiredTableSizes);

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->pConnection, c_pNetGPU->ppNeurons[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
		}

		cudaMemset(cnd_gpu->_send_num, 0, sizeof(int)*(cnd_gpu->_node_num));
		cudaGenerateCND<<<(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGP->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_g_idx2index, c_g_cross_index2idx, cnd_gpu->_send_data, cnd_gpu->_send_offset, cnd_gpu->_send_num, network->_nodeNum, time);

		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * network->_nodeNum, c_g_fired_n_num, sizeof(int)*network->_nodeNum, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(cnd->_send_num, cnd_gpu->_send_num, sizeof(int)*(cnd->_node_num), cudaMemcpyDeviceToHost));
		for (int i=0; i< network->_nodeNum; i++) {

			if (cnd->_send_[i] > 0) {
			}
		}

#ifdef LOG_DATA
		int currentIdx = time%(maxDelay+1);

		int copySize = 0;
		copyFromGPU<int>(&copySize, buffers->c_gFiredTableSizes + currentIdx, 1);
		if (copySize > 0) {
			copyFromGPU<int>(buffers->c_neuronsFired, buffers->c_gFiredTable + (allNeuronNum*currentIdx), copySize);
		}

		if (copy_idx >= 0 && (c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]) > 0) {
			copyFromGPU<real>(c_vm, c_g_vm, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		}
#endif

		for (int i=0; i<sTypeNum; i++) {
			assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->pConnection, c_pNetGPU->ppSynapses[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
		}

		for (int i=0; i< network->_nodeNum; i++) {
			// int i2idx = network->_nodeIdx + network->_nodeNum * i;
			// if (gCrossDataGPU->_firedNum[i2idx] > 0) {
			// 	int num = gCrossDataGPU->_firedNum[i2idx];
			// 	cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
			// }
		}
		
#ifdef LOG_DATA
		for (int i=0; i<copySize; i++) {
			fprintf(log_file, "%d ", buffers->c_neuronsFired[i]);
		}
		fprintf(log_file, "\n");

		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(v_file, "\n");
#endif
	}
	gettimeofday(&te, NULL);
	long seconds = te.tv_sec - ts.tv_sec;
	long hours = seconds/3600;
	seconds = seconds%3600;
	long minutes = seconds/60;
	seconds = seconds%60;
	long uSeconds = te.tv_usec - ts.tv_usec;
	if (uSeconds < 0) {
		uSeconds += 1000000;
		seconds = seconds - 1;
	}

	printf("Thread %d Simulation finesed in %ld:%ld:%ld.%06lds\n", network->_nodeIdx, hours, minutes, seconds, uSeconds);

	int *rate = (int*)malloc(sizeof(int)*nodeNeuronNum);
	copyFromGPU<int>(rate, buffers->c_gFireCount, nodeNeuronNum);

	char fire_filename[512];
	sprintf(fire_filename, "fire_%d.gpu.count", network->_nodeIdx); 
	FILE *rate_file = fopen(fire_filename, "w+");
	if (rate_file == NULL) {
		printf("Open file Sim.log failed\n");
		return 0;
	}

	for (int i=0; i<nodeNeuronNum; i++) {
		fprintf(rate_file, "%d \t", rate[i]);
	}

	free(rate);
	fclose(rate_file);

	fclose(log_file);
	fclose(v_file);

	free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);

	return 0;
}

int MultiNodeSimulator::run(real time, FireInfo &log)
{

	int sim_cycle = round(time/_dt);
	reset();

	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &node_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, node_id, node_num);

	SimInfo info(_dt);

	DistriNetwork *network = NULL;
	CrossNodeData *data = NULL;

	if (node_id == 0) {
		_network->setNodeNum(node_num);
		DistriNetwork *node_nets = _network->buildNetworks(info);
		CrossNodeData *node_datas = _network->arrangeNodeData(node_num);

		for (int i=0; i<node_num; i++) {
			node_nets[i]._simCycle = sim_cycle;
			node_nets[i]._nodeIdx = i;
			node_nets[i]._nodeNum = node_num;
			node_nets[i]._dt = _dt;
		}

		network = &(node_nets[0]);
		data = &(node_datas[0]);
		allocDataCND(data);

		for (int i=1; i<node_num; i++) {
			sendDistriNet(&(node_nets[i]), i, DATA_TAG, MPI_COMM_WORLD);
			sendCND(&(node_datas[i]), i, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
		}
	} else {
		network = recvDistriNet(0, DATA_TAG, MPI_COMM_WORLD);
		data = recvCND(0, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	run_node_gpu(network, data);

	return 0;
}

