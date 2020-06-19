/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "../gpu_utils/mem_op.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/GBuffers.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"

#include "MultiGPUSimulator.h"

using std::cout;
using std::endl;

pthread_barrier_t gpuCycleBarrier;

CrossNodeDataGPU * gCrossDataGPU;

MultiGPUSimulator::MultiGPUSimulator(Network *network, real dt) : Simulator(network, dt)
{
}

MultiGPUSimulator::~MultiGPUSimulator()
{
}

void *run_thread_gpu(void *para);

int MultiGPUSimulator::run(real time, FireInfo &log)
{
	int sim_cycle = round(time/_dt);
	reset();

	int device_count = 4;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	assert(device_count != 0);
	for (int i=0; i<device_count; i++) {
		for (int j=0; j<device_count; j++) {
			if (i!=j) {
				int access = 0;
				checkCudaErrors(cudaDeviceCanAccessPeer(&access, i, j));
				if (access == 1) {
					checkCudaErrors(cudaSetDevice(i));
					checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
				}
			}
		}
	}
	checkCudaErrors(cudaSetDevice(0));

	pthread_barrier_init(&gpuCycleBarrier, NULL, device_count);

	// MultiNetwork multiNet(_network, device_count);
	_network->setNodeNum(device_count);

	SimInfo info(_dt);
	DistriNetwork *node_nets = _network->buildNetworks(info);
	assert(node_nets != NULL);
	gCrossDataGPU = _network->arrangeCrossNodeDataGPU(device_count);
	assert(gCrossDataGPU != NULL);

	pthread_t *thread_ids = (pthread_t *)malloc(sizeof(pthread_t) * device_count);
	assert(thread_ids != NULL);


	for (int i=0; i<device_count; i++) {
		node_nets[i]._simCycle = sim_cycle;
		node_nets[i]._nodeIdx = i;
		node_nets[i]._nodeNum = device_count;
		node_nets[i]._dt = _dt;


		int ret = pthread_create(&(thread_ids[i]), NULL, &run_thread_gpu, (void*)&(node_nets[i]));
		assert(ret == 0);
	}

	for (int i=0; i<device_count; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	pthread_barrier_destroy(&gpuCycleBarrier);

	return 0;
}

void * run_thread_gpu(void *para) {
	DistriNetwork *network = (DistriNetwork*)para;

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
	GNetwork *c_pNetGPU = copyNetworkToGPU(pNetCPU);

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->pConnection->nNum;
	int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d NeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, nodeSynapseNum);

	//int dataOffset = network->_nodeIdx * network->_nodeNum;
	//int dataIdx = network->_nodeIdx * network->_nodeNum + network->_nodeIdx;

	int maxDelay = pNetCPU->pConnection->maxDelay;
	int minDelay = pNetCPU->pConnection->minDelay;
	// int deltaDelay = maxDelay - minDelay;
	// int deltaDelay = pNetCPU->pConnection->maxDelay - pNetCPU->pConnection->minDelay;
	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	// init_connection<<<1, 1>>>(c_pNetGPU->pConnection);

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

	//int * c_g_cross_id = gpuMalloc<int>(global_cross_data[dataIdx]._max_n_num); 

	int * c_g_idx2index = copyToGPU<int>(network->_crossnodeMap->_idx2index, allNeuronNum);
	int * c_g_cross_index2idx = copyToGPU<int>(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossSize);
	int * c_g_global_cross_data = gpuMalloc<int>(allNeuronNum * network->_nodeNum);
	int * c_g_fired_n_num = gpuMalloc<int>(network->_nodeNum);

	vector<int> firedInfo;
	struct timeval ts, te;
	//struct timeval t0, t1, t2, t3, t4, t5,/* t6,*/ t7, t8, t9;
	//double barrier1_time = 0, gpu_cpy_time = 0, peer_cpy_time = 0, barrier2_time=0, copy_time = 0;
	gettimeofday(&ts, NULL);
	for (int time=0; time<network->_simCycle; time++) {
		update_time<<<1, 1>>>(c_pNetGPU->pConnection, time, buffers->c_gFiredTableSizes);

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->pConnection, c_pNetGPU->ppNeurons[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
		}

		//gettimeofday(&t0, NULL);
		pthread_barrier_wait(&gpuCycleBarrier);
		//gettimeofday(&t1, NULL);
		//barrier1_time += (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)/1000000.0;
		cudaMemset(c_g_fired_n_num, 0, sizeof(int)*network->_nodeNum);
		//cudaDeviceSynchronize();
		//gettimeofday(&t2, NULL);
		//cudaDeliverNeurons(c_g_idx2index, c_g_cross_index2idx, c_g_global_cross_data, c_g_fired_n_num, network->_nodeNum, allNeuronNum);
		//for (int i=0; i<network->_nodeNum; i++) {
		//	int offset = i * network->_nodeNum + network->_nodeIdx; 
		//	copyFromGPU<int>(&(global_cross_data[offset]._fired_n_num), c_g_fired_n_num + i, 1);
		//	if (global_cross_data[offset]._fired_n_num > 0) {
		//		copyFromGPU<int>(global_cross_data[offset]._fired_n_idxs, c_g_global_cross_data + allNeuronNum * i, global_cross_data[offset]._fired_n_num);
		//	}
		//}

		cudaDeliverNeurons<<<(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_g_idx2index, c_g_cross_index2idx, c_g_global_cross_data, c_g_fired_n_num, network->_nodeNum, time);

		checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * network->_nodeNum, c_g_fired_n_num, sizeof(int)*network->_nodeNum, cudaMemcpyDeviceToHost));
		//gettimeofday(&t3, NULL);

		for (int i=0; i< network->_nodeNum; i++) {
			int idx2i = network->_nodeIdx * network->_nodeNum + i;
			assert(gCrossDataGPU->_firedNum[idx2i] <= gCrossDataGPU->_maxNum[idx2i]);
			if (gCrossDataGPU->_firedNum[idx2i] > 0) {
				checkCudaErrors(cudaMemcpyPeer(gCrossDataGPU->_firedArrays[idx2i], i, c_g_global_cross_data + allNeuronNum * i, network->_nodeIdx, gCrossDataGPU->_firedNum[idx2i] * sizeof(int)));
			}
		}
		//gettimeofday(&t7, NULL);

		//gpu_cpy_time += (t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec)/1000000.0;
		//peer_cpy_time += (t7.tv_sec - t3.tv_sec) + (t7.tv_usec - t3.tv_usec)/1000000.0;

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
		//cudaDeviceSynchronize();

		//gettimeofday(&t4, NULL);
		pthread_barrier_wait(&gpuCycleBarrier);
		//gettimeofday(&t5, NULL);
		//barrier2_time += (t5.tv_sec - t4.tv_sec) + (t5.tv_usec - t4.tv_usec)/1000000.0;

		//gettimeofday(&t6, NULL);
		//collectNeurons();
		//gettimeofday(&t7, NULL);
		//cpu_cpy_time += (t7.tv_sec - t6.tv_sec) + (t7.tv_usec - t6.tv_usec)/1000000.0;
		
		//gettimeofday(&t8, NULL);
		//if (global_cross_data[dataIdx]._fired_n_num > 0) {
		//	copyToGPU(c_g_cross_id, global_cross_data[dataIdx]._fired_n_idxs, global_cross_data[dataIdx]._fired_n_num);
		//	addCrossNeurons(c_g_cross_id, global_cross_data[dataIdx]._fired_n_num);
		//}
		for (int i=0; i< network->_nodeNum; i++) {
			int i2idx = network->_nodeIdx + network->_nodeNum * i;
			if (gCrossDataGPU->_firedNum[i2idx] > 0) {
				int num = gCrossDataGPU->_firedNum[i2idx];
				cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
			}
		}
		
		//gettimeofday(&t9, NULL);
		//copy_time += (t9.tv_sec - t8.tv_sec) + (t9.tv_usec - t8.tv_usec)/1000000.0;

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

		//pthread_barrier_wait(&gpuCycleBarrier);
	}
	pthread_barrier_wait(&gpuCycleBarrier);
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
	//printf("Thread %d cost : barrier1 %lf, DtoH %lf, DtoD %lf, barrier2 %lf, HtoD %lf\n", network->_nodeIdx, barrier1_time, gpu_cpy_time, peer_cpy_time, barrier2_time, copy_time);

	int *rate = (int*)malloc(sizeof(int)*nodeNeuronNum);
	copyFromGPU<int>(rate, buffers->c_gFireCount, nodeNeuronNum);

	char fire_filename[512];
	sprintf(fire_filename, "fire_%d.gpu.count", network->_nodeIdx); 
	FILE *rate_file = fopen(fire_filename, "w+");
	if (rate_file == NULL) {
		printf("Open file Sim.log failed\n");
		return NULL;
	}

	for (int i=0; i<nodeNeuronNum; i++) {
		fprintf(rate_file, "%d \t", rate[i]);
	}

	free(rate);
	fclose(rate_file);

	fclose(log_file);
	fclose(v_file);

	free_buffers(buffers);
	freeNetworkGPU(c_pNetGPU);

	return NULL;
}


int MultiGPUSimulator::run_single(real time, FireInfo &log)
{
	int sim_cycle = round(time/_dt);
	reset();

	int device_count = 4;

	checkCudaErrors(cudaSetDevice(0));

	_network->setNodeNum(device_count);

	SimInfo info(_dt);
	DistriNetwork *node_nets = _network->buildNetworks(info);
	assert(node_nets != NULL);
	gCrossDataGPU = _network->arrangeCrossNodeDataGPU(device_count);
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
	int **c_g_idx2index = (int **)malloc(sizeof(int*)*device_count);
	int **c_g_cross_index2idx = (int **)malloc(sizeof(int*)*device_count);
	int **c_g_global_cross_data = (int **)malloc(sizeof(int*)*device_count);
	int **c_g_fired_n_num = (int **)malloc(sizeof(int*)*device_count);

	real **c_vm = (real **)malloc(sizeof(real*)*device_count);
	real **c_g_vm = (real **)malloc(sizeof(real*)*device_count);
	int *copy_idx = (int *)malloc(sizeof(int)*device_count);

	char log_filename[512];
	char v_filename[512];
	for (int d=0; d<device_count; d++) {
		sprintf(log_filename, "sim_%d.log", node_nets[d]._nodeIdx); 
		log_file[d] = fopen(log_filename, "w+");
		assert(log_file[d] != NULL);

		sprintf(v_filename, "v_%d.data", node_nets[d]._nodeIdx); 
		v_file[d] = fopen(v_filename, "w+");
		assert(v_file != NULL);

		GNetwork *pNetCPU = node_nets[d]._network;
		c_pNetGPU[d] = copyNetworkToGPU(pNetCPU);

		int nTypeNum = c_pNetGPU[d]->nTypeNum;
		int sTypeNum = c_pNetGPU[d]->sTypeNum;
		int nodeNeuronNum = c_pNetGPU[d]->pNeuronNums[nTypeNum];
		int allNeuronNum = c_pNetGPU[d]->pConnection->nNum;
		int nodeSynapseNum = c_pNetGPU[d]->pSynapseNums[sTypeNum];
		printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", node_nets[d]._nodeIdx, nTypeNum, sTypeNum);
		printf("Thread %d NeuronNum: %d, SynapseNum: %d\n", node_nets[d]._nodeIdx, nodeNeuronNum, nodeSynapseNum);

		int maxDelay = pNetCPU->pConnection->maxDelay;
		int minDelay = pNetCPU->pConnection->minDelay;
		printf("Thread %d MaxDelay: %d MinDelay: %d\n", node_nets[d]._nodeIdx, maxDelay,  minDelay);

		buffers[d] = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->pConnection->maxDelay, node_nets[d]._dt);

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

		c_g_idx2index[d] = copyToGPU<int>(node_nets[d]._crossnodeMap->_idx2index, allNeuronNum);
		c_g_cross_index2idx[d] = copyToGPU<int>(node_nets[d]._crossnodeMap->_crossnodeIndex2idx, node_nets[d]._crossnodeMap->_crossSize);
		c_g_global_cross_data[d] = gpuMalloc<int>(allNeuronNum * node_nets[d]._nodeNum);
		c_g_fired_n_num[d] = gpuMalloc<int>(node_nets[d]._nodeNum);
	}


	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);
	for (int time=0; time<node_nets[0]._simCycle; time++) {
		for (int d=0; d<device_count; d++) {
			update_time<<<1, 1>>>(c_pNetGPU[d]->pConnection, time, buffers[d]->c_gFiredTableSizes);

			for (int i=0; i<c_pNetGPU[d]->nTypeNum; i++) {
				assert(c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i] > 0);
				cudaUpdateType[c_pNetGPU[d]->pNTypes[i]](c_pNetGPU[d]->pConnection, c_pNetGPU[d]->ppNeurons[i], buffers[d]->c_gNeuronInput, buffers[d]->c_gNeuronInput_I, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_pNetGPU[d]->pNeuronNums[i+1]-c_pNetGPU[d]->pNeuronNums[i], c_pNetGPU[d]->pNeuronNums[i], time, &updateSize[d][c_pNetGPU[d]->pNTypes[i]]);
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
			cudaDeliverNeurons<<<(c_pNetGPU[d]->pConnection->nNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU[d]->pConnection, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_g_idx2index[d], c_g_cross_index2idx[d], c_g_global_cross_data[d], c_g_fired_n_num[d], node_nets[d]._nodeNum, time);

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
				cudaUpdateType[c_pNetGPU[d]->pSTypes[i]](c_pNetGPU[d]->pConnection, c_pNetGPU[d]->ppSynapses[i], buffers[d]->c_gNeuronInput, buffers[d]->c_gNeuronInput_I, buffers[d]->c_gFiredTable, buffers[d]->c_gFiredTableSizes, c_pNetGPU[d]->pSynapseNums[i+1]-c_pNetGPU[d]->pSynapseNums[i], c_pNetGPU[d]->pSynapseNums[i], time, &updateSize[d][c_pNetGPU[d]->pSTypes[i]]);
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
		sprintf(fire_filename, "fire_%d.count", node_nets[d]._nodeIdx); 
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
		freeNetworkGPU(c_pNetGPU[d]);
	}

	return 0;
}
