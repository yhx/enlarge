/* This program is writen by qp09.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>

#include "../utils/utils.h"
#include "../base/TypeFunc.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "../gpu_utils/runtime.h"
//#include "../gpu_utils/GBuffers.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"

#include "MultiGPUSimulator.h"

using std::cout;
using std::endl;

pthread_barrier_t gpuCycleBarrier;

CrossThreadDataGPU * gCrossDataGPU;

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
	_network->set_node_num(device_count);

	SimInfo info(_dt);
	DistriNetwork *node_nets = _network->buildNetworks(info);
	assert(node_nets != NULL);
	gCrossDataGPU = _network->arrangeCrossGPUData();
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
	sprintf(v_filename, "v_%d.gpu.log", network->_nodeIdx); 
	FILE *v_file = fopen(v_filename, "w+");
	assert(v_file != NULL);

	checkCudaErrors(cudaSetDevice(network->_nodeIdx));

	GNetwork *pNetCPU = network->_network;
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
	int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d NeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, nodeSynapseNum);

	//int dataOffset = network->_nodeIdx * network->_nodeNum;
	//int dataIdx = network->_nodeIdx * network->_nodeNum + network->_nodeIdx;

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	int minDelay = pNetCPU->ppConnections[0]->minDelay;
	// int deltaDelay = maxDelay - minDelay;
	// int deltaDelay = pNetCPU->pConnection->maxDelay - pNetCPU->pConnection->minDelay;
	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	// init_connection<<<1, 1>>>(c_pNetGPU->pConnection);

	// GBuffers *buffers = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->ppConnections[0]->maxDelay, network->_dt);
	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, maxDelay, network->_nodeIdx);
	Buffer *g_buffer = buffer._gpu_array;

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
	CrossNodeMap *cnm_gpu = to_gpu(network->_crossnodeMap);

	// integer_t * c_g_idx2index = copyToGPU(network->_crossnodeMap->_idx2index, allNeuronNum);
	// integer_t * c_g_cross_index2idx = copyToGPU(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossSize);
	uinteger_t * c_g_global_cross_data = gpuMalloc<uinteger_t>(allNeuronNum * network->_nodeNum);
	uinteger_t * c_g_fired_n_num = gpuMalloc<uinteger_t>(network->_nodeNum);

	vector<int> firedInfo;
	struct timeval ts, te;
	//struct timeval t0, t1, t2, t3, t4, t5,/* t6,*/ t7, t8, t9;
	//double barrier1_time = 0, gpu_cpy_time = 0, peer_cpy_time = 0, barrier2_time=0, copy_time = 0;
	gettimeofday(&ts, NULL);
	for (int time=0; time<network->_simCycle; time++) {
		update_time<<<1, 1>>>(g_buffer->_fired_sizes, maxDelay, time);

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppNeurons[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
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

		cudaDeliverNeurons<<<(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(g_buffer->_fire_table, g_buffer->_fired_sizes, cnm_gpu->_idx2index, cnm_gpu->_crossnodeIndex2idx, c_g_global_cross_data, c_g_fired_n_num, maxDelay, network->_nodeNum, time);

		copyFromGPU(gCrossDataGPU->_firedNum + network->_nodeIdx * network->_nodeNum, c_g_fired_n_num, network->_nodeNum);
		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * network->_nodeNum, c_g_fired_n_num, sizeof(int)*network->_nodeNum, cudaMemcpyDeviceToHost));
		//gettimeofday(&t3, NULL);

		for (int i=0; i< network->_nodeNum; i++) {
			int idx2i = network->_nodeIdx * network->_nodeNum + i;
			assert(gCrossDataGPU->_firedNum[idx2i] <= gCrossDataGPU->_maxNum[idx2i]);
			if (gCrossDataGPU->_firedNum[idx2i] > 0) {
				gpuMemcpyPeer(gCrossDataGPU->_firedArrays[idx2i], i, c_g_global_cross_data + allNeuronNum * i, network->_nodeIdx, gCrossDataGPU->_firedNum[idx2i]);
			}
		}
		//gettimeofday(&t7, NULL);

		//gpu_cpy_time += (t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec)/1000000.0;
		//peer_cpy_time += (t7.tv_sec - t3.tv_sec) + (t7.tv_usec - t3.tv_usec)/1000000.0;

#ifdef LOG_DATA
		int currentIdx = time%(maxDelay+1);

		uinteger_t copySize = 0;
		copyFromGPU(&copySize, g_buffer->_fired_sizes + currentIdx, 1);
		assert(copySize <= allNeuronNum);
		if (copySize > 0) {
			copyFromGPU(buffer._fire_table, g_buffer->_fire_table + (allNeuronNum*currentIdx), copySize);
		}

		if (copy_idx >= 0 && (c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]) > 0) {
			copyFromGPU(c_vm, c_g_vm, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		}
#endif

		for (int i=0; i<sTypeNum; i++) {
			assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
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
				size_t num = gCrossDataGPU->_firedNum[i2idx];
				cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(g_buffer->_fire_table, g_buffer->_fired_sizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], maxDelay, time);
			}
		}
		
		//gettimeofday(&t9, NULL);
		//copy_time += (t9.tv_sec - t8.tv_sec) + (t9.tv_usec - t8.tv_usec)/1000000.0;

#ifdef LOG_DATA
		for (int i=0; i<copySize; i++) {
			fprintf(log_file, "%d ", buffer._fire_table[i]);
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

	char fire_filename[512];
	sprintf(fire_filename, "gpu_%d", network->_nodeIdx); 

	for (int i=0; i<nTypeNum; i++) {
		cudaLogRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], c_pNetGPU->ppNeurons[i],  fire_filename);
	}

	fclose(log_file);
	fclose(v_file);

	// free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);

	return NULL;
}

