
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "../gpu_utils/helper_gpu.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/GBuffers.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"

#include "MultiGPUSimulator.h"

using std::cout;
using std::endl;

int MultiGPUSimulator::run_single(real time)
{
	int sim_cycle = round(time/_dt);
	reset();

	int device_count = 4;

	checkCudaErrors(cudaSetDevice(0));

	_network->set_node_num(device_count);

	SimInfo info(_dt);
	DistriNetwork *node_nets = _network->buildNetworks(info);
	assert(node_nets != NULL);
	CrossThreadDataGPU *gCrossDataGPU = _network->arrangeCrossThreadDataGPU(device_count);
	assert(gCrossDataGPU != NULL);


	for (int d=0; d<device_count; d++) {
		node_nets[d]._simCycle = sim_cycle;
		node_nets[d]._nodeIdx = d;
		node_nets[d]._nodeNum = device_count;
		node_nets[d]._dt = _dt;
	}


	FILE **log_file = (FILE **)malloc(sizeof(FILE*)*device_count);
	FILE **v_file = (FILE **)malloc(sizeof(FILE*)*device_count);

	GNetwork **c_pNetGPU =  (GNetwork **)malloc(sizeof(GNetwork*)*device_count);

	GBuffers **buffers = (GBuffers **)malloc(sizeof(GBuffers*)*device_count);

	BlockSize **updateSize = (BlockSize **)malloc(sizeof(BlockSize*)*device_count);
	size_t **c_g_idx2index = (size_t **)malloc(sizeof(size_t*)*device_count);
	size_t **c_g_cross_index2idx = (size_t **)malloc(sizeof(size_t*)*device_count);
	size_t **c_g_global_cross_data = (size_t **)malloc(sizeof(size_t*)*device_count);
	size_t **c_g_fired_n_num = (size_t **)malloc(sizeof(size_t*)*device_count);

	real **c_vm = (real **)malloc(sizeof(real*)*device_count);
	real **c_g_vm = (real **)malloc(sizeof(real*)*device_count);
	int *copy_idx = (int *)malloc(sizeof(int)*device_count);

	char log_filename[512];
	char v_filename[512];
	for (int d=0; d<device_count; d++) {
		sprintf(log_filename, "sim_%d.gpu.log", node_nets[d]._nodeIdx); 
		log_file[d] = fopen(log_filename, "w+");
		assert(log_file[d] != NULL);

		sprintf(v_filename, "v_%d.gpu.data", node_nets[d]._nodeIdx); 
		v_file[d] = fopen(v_filename, "w+");
		assert(v_file != NULL);

		GNetwork *pNetCPU = node_nets[d]._network;
		c_pNetGPU[d] = copyGNetworkToGPU(pNetCPU);

		int nTypeNum = c_pNetGPU[d]->nTypeNum;
		int sTypeNum = c_pNetGPU[d]->sTypeNum;
		int nodeNeuronNum = c_pNetGPU[d]->pNeuronNums[nTypeNum];
		int allNeuronNum = c_pNetGPU[d]->ppConnections[0]->nNum;
		int nodeSynapseNum = c_pNetGPU[d]->pSynapseNums[sTypeNum];
		printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", node_nets[d]._nodeIdx, nTypeNum, sTypeNum);
		printf("Thread %d NeuronNum: %d, SynapseNum: %d\n", node_nets[d]._nodeIdx, nodeNeuronNum, nodeSynapseNum);

		int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
		int minDelay = pNetCPU->ppConnections[0]->minDelay;
		printf("Thread %d MaxDelay: %d MinDelay: %d\n", node_nets[d]._nodeIdx, maxDelay,  minDelay);

		buffers[d] = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->ppConnections[0]->maxDelay, node_nets[d]._dt);

		updateSize[d] = getBlockSize(allNeuronNum, nodeSynapseNum);

#ifdef LOG_DATA
		c_vm[d] = hostMalloc<real>(nodeNeuronNum);
		int life_idx = getIndex(c_pNetGPU[d]->pNTypes, nTypeNum, LIF);
		copy_idx[d] = -1;

		if (life_idx >= 0) {
			LIFData *c_g_lif = copyFromGPU<LIFData>(static_cast<LIFData *>(c_pNetGPU[d]->ppNeurons[life_idx]), 1);
			c_g_vm[d] = c_g_lif->pV_m;
			copy_idx[d] = life_idx;
		} else {
		}
#endif

		for (int i=0; i<nTypeNum; i++) {
			cout << "Thread " << node_nets[d]._nodeIdx << " " << c_pNetGPU[d]->pNTypes[i] << ": <<<" << updateSize[d][c_pNetGPU[d]->pNTypes[i]].gridSize << ", " << updateSize[d][c_pNetGPU[d]->pNTypes[i]].blockSize << ">>>" << endl;
		}
		for (int i=0; i<sTypeNum; i++) {
			cout << "Thread " << node_nets[d]._nodeIdx << " " << c_pNetGPU[d]->pSTypes[i] << ": <<<" << updateSize[d][c_pNetGPU[d]->pSTypes[i]].gridSize << ", " << updateSize[d][c_pNetGPU[d]->pSTypes[i]].blockSize << ">>>" << endl;
		}

		c_g_idx2index[d] = copyToGPU<size_t>(node_nets[d]._crossnodeMap->_idx2index, allNeuronNum);
		c_g_cross_index2idx[d] = copyToGPU<size_t>(node_nets[d]._crossnodeMap->_crossnodeIndex2idx, node_nets[d]._crossnodeMap->_crossSize);
		c_g_global_cross_data[d] = gpuMalloc<size_t>(allNeuronNum * node_nets[d]._nodeNum);
		c_g_fired_n_num[d] = gpuMalloc<size_t>(node_nets[d]._nodeNum);
	}


	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);
	for (int time=0; time<node_nets[0]._simCycle; time++) {
		for (int d=0; d<device_count; d++) {
			int maxDelay = node_nets[d]._network->ppConnections[0]->maxDelay;
			update_time<<<1, 1>>>(buffers[d]->c_gFiredTableSizes, maxDelay, time);

			for (int i=0; i<c_pNetGPU[d]->nTypeNum; i++) {
				assert(c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i] > 0);
				cudaUpdateType[c_pNetGPU[d]->pNTypes[i]](c_pNetGPU[d]->ppConnections[0], c_pNetGPU[d]->ppNeurons[i], buffers[d]->c_gNeuronInput, buffers[d]->c_gNeuronInput_I, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i], c_pNetGPU[d]->pNeuronNums[i], time, &updateSize[d][c_pNetGPU[d]->pNTypes[i]]);
			}

			cudaMemset(c_g_fired_n_num[d], 0, sizeof(int)*node_nets[d]._nodeNum);
		}
		cudaDeviceSynchronize();
		//cudaDeliverNeurons(c_g_idx2index, c_g_cross_index2idx, c_g_global_cross_data, c_g_fired_n_num, node_nets[d]._nodeNum, allNeuronNum);
		//for (int i=0; i<node_nets[d]._nodeNum; i++) {
		//	int offset = i * node_nets[d]._nodeNum + node_nets[d]._nodeIdx; 
		//	copyFromGPU<int>(&(global_cross_data[offset]._fired_n_num), c_g_fired_n_num + i, 1);
		//	if (global_cross_data[offset]._fired_n_num > 0) {
		//		copyFromGPU<int>(global_cross_data[offset]._fired_n_idxs, c_g_global_cross_data + allNeuronNum * i, global_cross_data[offset]._fired_n_num);
		//	}
		//}

		for (int d=0; d<device_count; d++) {
			cudaDeliverNeurons<<<(c_pNetGPU[d]->pConnection->nNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_g_idx2index[d], c_g_cross_index2idx[d], c_g_global_cross_data[d], c_g_fired_n_num[d], maxDelay, node_nets[d]._nodeNum, time);

			checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + node_nets[d]._nodeIdx * node_nets[d]._nodeNum, c_g_fired_n_num[d], sizeof(int)*node_nets[d]._nodeNum, cudaMemcpyDeviceToHost));
		}

		for (int d=0; d<device_count; d++) {
			for (int i=0; i< node_nets[d]._nodeNum; i++) {
				int idx2i = node_nets[d]._nodeIdx * node_nets[d]._nodeNum + i;
				assert(gCrossDataGPU->_firedNum[idx2i] <= gCrossDataGPU->_maxNum[idx2i]);
				if (gCrossDataGPU->_firedNum[idx2i] > 0) {
					checkCudaErrors(cudaMemcpyPeer(gCrossDataGPU->_firedArrays[idx2i], i, c_g_global_cross_data[d] + c_pNetGPU[d]->pConnection->nNum * i, node_nets[d]._nodeIdx, gCrossDataGPU->_firedNum[idx2i] * sizeof(int)));
				}
			}
		}

#ifdef LOG_DATA
		int *copySize = (int *)malloc(sizeof(int) * device_count);
		for (int d=0; d<device_count; d++) {
			int currentIdx = time%(c_pNetGPU[d]->pConnection->maxDelay+1);

			copyFromGPU<int>(&copySize[d], buffers[d]->c_gFiredTableSizes + currentIdx, 1);
			if (copySize[d] > 0) {
				copyFromGPU<int>(buffers[d]->c_neuronsFired, buffers[d]->c_gFiredTable + (c_pNetGPU[d]->pConnection->nNum*currentIdx), copySize[d]);
			}

			if (copy_idx[d] >= 0 && (c_pNetGPU[d]->pNeuronNums[copy_idx[d]+1]-c_pNetGPU[d]->pNeuronNums[copy_idx[d]]) > 0) {
				copyFromGPU<real>(c_vm[d], c_g_vm[d], c_pNetGPU[d]->pNeuronNums[copy_idx[d]+1]-c_pNetGPU[d]->pNeuronNums[copy_idx[d]]);
			}
		}
#endif

		for (int d=0; d<device_count; d++) {
			for (int i=0; i<c_pNetGPU[d]->sTypeNum; i++) {
				assert(c_pNetGPU[d]->pSynapseNums[i+1]-c_pNetGPU[d]->pSynapseNums[i] > 0);
				cudaUpdateType[c_pNetGPU[d]->pSTypes[i]](c_pNetGPU[d]->ppConnections[i], c_pNetGPU[d]->ppSynapses[i], buffers[d]->c_gNeuronInput, buffers[d]->c_gNeuronInput_I, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_pNetGPU[d]->pSynapseNums[i+1]-c_pNetGPU[d]->pSynapseNums[i], c_pNetGPU[d]->pSynapseNums[i], time, &updateSize[d][c_pNetGPU[d]->pSTypes[i]]);
			}
		}		
		cudaDeviceSynchronize();

		//collectNeurons();
		//if (global_cross_data[dataIdx]._fired_n_num > 0) {
		//	copyToGPU(c_g_cross_id, global_cross_data[dataIdx]._fired_n_idxs, global_cross_data[dataIdx]._fired_n_num);
		//	addCrossNeurons(c_g_cross_id, global_cross_data[dataIdx]._fired_n_num);
		//}
		for (int d=0; d<device_count; d++) {
			for (int i=0; i< node_nets[d]._nodeNum; i++) {
				int i2idx = node_nets[d]._nodeIdx + node_nets[d]._nodeNum * i;
				if (gCrossDataGPU->_firedNum[i2idx] > 0) {
					int num = gCrossDataGPU->_firedNum[i2idx];
					cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU[d]->pConnection, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
				}
			}
		}
		
#ifdef LOG_DATA
		for (int d=0; d<device_count; d++) {
			for (int i=0; i<copySize[d]; i++) {
				fprintf(log_file[d], "%d ", buffers[d]->c_neuronsFired[i]);
			}
			fprintf(log_file[d], "\n");

			for (int i=0; i<c_pNetGPU[d]->pNeuronNums[copy_idx[d]+1] - c_pNetGPU[d]->pNeuronNums[copy_idx[d]]; i++) {
				fprintf(v_file[d], "%.10lf \t", c_vm[d][i]);
			}
			fprintf(v_file[d], "\n");
		}
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

	printf("Simulation finesed in %ld:%ld:%ld.%06lds\n", hours, minutes, seconds, uSeconds);

	for (int d=0; d<device_count; d++) {
		int nodeNeuronNum = c_pNetGPU[d]->pNeuronNums[c_pNetGPU[d]->nTypeNum];
		int *rate = (int*)malloc(sizeof(int)*nodeNeuronNum);
		copyFromGPU<int>(rate, buffers[d]->c_gFireCount, nodeNeuronNum);

		char fire_filename[512];
		sprintf(fire_filename, "fire_%d.gpu.count", node_nets[d]._nodeIdx); 
		FILE *rate_file = fopen(fire_filename, "w+");
		if (rate_file == NULL) {
			printf("Open file Sim.log failed\n");
			return -1;
		}

		for (int i=0; i<nodeNeuronNum; i++) {
			fprintf(rate_file, "%d \t", rate[i]);
		}
		free(rate);
		fclose(rate_file);
	}

	for (int d=0; d<device_count; d++) {
		fclose(log_file[d]);
		fclose(v_file[d]);

		free_buffers(buffers[d]);
		freeGNetworkGPU(c_pNetGPU[d]);
	}

	return 0;
}
