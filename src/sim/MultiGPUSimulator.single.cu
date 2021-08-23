
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>

#include "../base/TypeFunc.h"
#include "../utils/utils.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "../gpu_utils/runtime.h"
// #include "../gpu_utils/GBuffers.h"
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
	CrossThreadDataGPU *gCrossDataGPU = _network->arrangeCrossGPUData();
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

	// GBuffers **buffers = (GBuffers **)malloc(sizeof(GBuffers*)*device_count);
	Buffer **buffers = new Buffer*[device_count]();

	BlockSize **updateSize = (BlockSize **)malloc(sizeof(BlockSize*)*device_count);
	integer_t **c_g_idx2index = malloc_c<integer_t*>(device_count);
	integer_t **c_g_cross_index2idx = malloc_c<integer_t*>(device_count);
	uinteger_t **c_g_global_cross_data = malloc_c<uinteger_t*>(device_count);
	uinteger_t **c_g_fired_n_num = malloc_c<uinteger_t*>(device_count);

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

		// buffers[d] = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->ppConnections[0]->maxDelay, node_nets[d]._dt);
		buffers[d] = new Buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, pNetCPU->ppConnections[0]->maxDelay, 0);

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

		c_g_idx2index[d] = copyToGPU(node_nets[d]._crossnodeMap->_idx2index, allNeuronNum);
		c_g_cross_index2idx[d] = copyToGPU(node_nets[d]._crossnodeMap->_crossnodeIndex2idx, node_nets[d]._crossnodeMap->_crossSize);
		c_g_global_cross_data[d] = gpuMalloc<uinteger_t>(allNeuronNum * node_nets[d]._nodeNum);
		c_g_fired_n_num[d] = gpuMalloc<uinteger_t>(node_nets[d]._nodeNum);
	}


	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);
	for (int time=0; time<node_nets[0]._simCycle; time++) {
		for (int d=0; d<device_count; d++) {
			int allNeuronNum = node_nets[d]._network->ppConnections[0]->nNum;
			int maxDelay = node_nets[d]._network->ppConnections[0]->maxDelay;
			update_time<<<1, 1>>>(buffers[d]->_fired_sizes, maxDelay, time);

			for (int i=0; i<c_pNetGPU[d]->nTypeNum; i++) {
				assert(c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i] > 0);
				cudaUpdateType[c_pNetGPU[d]->pNTypes[i]](c_pNetGPU[d]->ppConnections[i], c_pNetGPU[d]->ppNeurons[i], buffers[d]->_data, buffers[d]->_fire_table, buffers[d]->_fired_sizes, allNeuronNum, c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i], c_pNetGPU[d]->pNeuronNums[i], time, &updateSize[d][c_pNetGPU[d]->pNTypes[i]]);
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
			int allNeuronNum = node_nets[d]._network->ppConnections[0]->nNum;
			int maxDelay = node_nets[d]._network->ppConnections[0]->maxDelay;
			cudaDeliverNeurons<<<(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(buffers[d]->_fire_table, buffers[d]->_fired_sizes, c_g_idx2index[d], c_g_cross_index2idx[d], c_g_global_cross_data[d], c_g_fired_n_num[d], maxDelay, node_nets[d]._nodeNum, time);

			checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + node_nets[d]._nodeIdx * node_nets[d]._nodeNum, c_g_fired_n_num[d], sizeof(int)*node_nets[d]._nodeNum, cudaMemcpyDeviceToHost));
		}

		for (int d=0; d<device_count; d++) {
			for (int i=0; i< node_nets[d]._nodeNum; i++) {
				int idx2i = node_nets[d]._nodeIdx * node_nets[d]._nodeNum + i;
				assert(gCrossDataGPU->_firedNum[idx2i] <= gCrossDataGPU->_maxNum[idx2i]);
				if (gCrossDataGPU->_firedNum[idx2i] > 0) {
					gpuMemcpyPeer(gCrossDataGPU->_firedArrays[idx2i], i, c_g_global_cross_data[d] + c_pNetGPU[d]->ppConnections[0]->nNum * i, node_nets[d]._nodeIdx, gCrossDataGPU->_firedNum[idx2i]);
				}
			}
		}

#ifdef LOG_DATA
		uinteger_t *copySize = malloc_c<uinteger_t>(device_count);
		for (int d=0; d<device_count; d++) {
			int currentIdx = time%(c_pNetGPU[d]->ppConnections[0]->maxDelay+1);

			copyFromGPU(&copySize[d], buffers[d]->_fired_sizes + currentIdx, 1);
			assert(copySize[d] <= c_pNetGPU[d]->ppConnections[0]->nNum);
			if (copySize[d] > 0) {
				copyFromGPU(buffers[d]->_fire_table, buffers[d]->_fire_table + (c_pNetGPU[d]->ppConnections[0]->nNum*currentIdx), copySize[d]);
			}

			if (copy_idx[d] >= 0 && (c_pNetGPU[d]->pNeuronNums[copy_idx[d]+1]-c_pNetGPU[d]->pNeuronNums[copy_idx[d]]) > 0) {
				copyFromGPU(c_vm[d], c_g_vm[d], c_pNetGPU[d]->pNeuronNums[copy_idx[d]+1]-c_pNetGPU[d]->pNeuronNums[copy_idx[d]]);
			}
		}
#endif

		for (int d=0; d<device_count; d++) {
			for (int i=0; i<c_pNetGPU[d]->sTypeNum; i++) {
				int allNeuronNum = node_nets[d]._network->ppConnections[0]->nNum;
				assert(c_pNetGPU[d]->pSynapseNums[i+1]-c_pNetGPU[d]->pSynapseNums[i] > 0);
				cudaUpdateType[c_pNetGPU[d]->pSTypes[i]](c_pNetGPU[d]->ppConnections[i], c_pNetGPU[d]->ppSynapses[i], buffers[d]->_data, buffers[d]->_fire_table, buffers[d]->_fired_sizes, allNeuronNum, c_pNetGPU[d]->pSynapseNums[i+1]-c_pNetGPU[d]->pSynapseNums[i], c_pNetGPU[d]->pSynapseNums[i], time, &updateSize[d][c_pNetGPU[d]->pSTypes[i]]);
			}
		}		
		cudaDeviceSynchronize();

		//collectNeurons();
		//if (global_cross_data[dataIdx]._fired_n_num > 0) {
		//	copyToGPU(c_g_cross_id, global_cross_data[dataIdx]._fired_n_idxs, global_cross_data[dataIdx]._fired_n_num);
		//	addCrossNeurons(c_g_cross_id, global_cross_data[dataIdx]._fired_n_num);
		//}
		for (int d=0; d<device_count; d++) {
			int maxDelay = node_nets[d]._network->ppConnections[0]->maxDelay;
			for (int i=0; i< node_nets[d]._nodeNum; i++) {
				int i2idx = node_nets[d]._nodeIdx + node_nets[d]._nodeNum * i;
				if (gCrossDataGPU->_firedNum[i2idx] > 0) {
					int num = gCrossDataGPU->_firedNum[i2idx];
					cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(buffers[d]->_fire_table, buffers[d]->_fired_sizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], maxDelay, time);
				}
			}
		}
		
#ifdef LOG_DATA
		for (int d=0; d<device_count; d++) {
			for (int i=0; i<copySize[d]; i++) {
				fprintf(log_file[d], "%d ", buffers[d]->_fire_table[i]);
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

		char fire_filename[512];
		sprintf(fire_filename, "gpu_%d", d); 

		GNetwork * net = node_nets[d]._network;

		for (int i=0; i<net->nTypeNum; i++) {
			cudaLogRateNeuron[net->pNTypes[i]](net->ppNeurons[i], c_pNetGPU[d]->ppNeurons[i], fire_filename);
		}
	}

	for (int d=0; d<device_count; d++) {
		fclose(log_file[d]);
		fclose(v_file[d]);

		// free_buffers(buffers[d]);
		delete buffers[d];
		freeGNetworkGPU(c_pNetGPU[d]);
	}
	delete buffers;

	return 0;
}
