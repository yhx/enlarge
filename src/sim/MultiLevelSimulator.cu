
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <mpi.h>

#include "../utils/utils.h"
// #include "../utils/helper_c.h"
#include "../base/TypeFunc.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "../../msg_utils/msg_utils/msg_utils.h"
#include "../../msg_utils/msg_utils/GPUManager.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/gpu_utils.h"
#include "MultiLevelSimulator.h"
// #include "../gpu_utils/GBuffers.h"

pthread_barrier_t g_proc_barrier;

void * run_thread_ml(void *para) {
	RunPara * tmp = static_cast<RunPara*>(para);
	DistriNetwork *network = tmp->_net;
	CrossMap *cm = tmp->_cm;
	CrossSpike **cs = tmp->_cs;
	int thread_id = tmp->_thread_id;
	// print_mem("Inside Run");

	FILE *v_file = log_file_mpi("v", network->_nodeIdx);
	FILE *sim_file = log_file_mpi("sim", network->_nodeIdx);
#ifdef LOG_DATA
	FILE *msg_file = log_file_mpi("msg", network->_nodeIdx);
	FILE *send_file = log_file_mpi("send", network->_nodeIdx);
	FILE *recv_file = log_file_mpi("recv", network->_nodeIdx);
	FILE *input_i_file = log_file_mpi("input_i", network->_nodeIdx);
	FILE *input_e_file = log_file_mpi("input_e", network->_nodeIdx);
#endif

	gm.set(thread_id);
	gm.lock();

	GNetwork *pNetCPU = network->_network;
	print_gmem("Before using GPU");
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);
	// print_mem("Copied Network");

	cm->to_gpu();
	cs[thread_id]->to_gpu();
	// print_mem("Copied CND");

	int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
	int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
	int nodeSynapseNum = pNetCPU->pSynapseNums[sTypeNum];

	int max_delay = pNetCPU->ppConnections[0]->maxDelay;
	int min_delay = pNetCPU->ppConnections[0]->minDelay;

	pInfoGNetwork(pNetCPU, string("Proc ") + std::to_string(network->_nodeIdx));


	//TODO Set GPU0 directly 
	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, max_delay, thread_id);
	Buffer *g_buffer = buffer._gpu_array;

	int *c_fired_sizes = hostMalloc<int>(max_delay+1);

	BlockSize *updateSize = getBlockSize(allNeuronNum, nodeSynapseNum);

	// print_mem("Alloced Buffers");

#ifdef LOG_DATA
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	real *c_vm = hostMalloc<real>(nodeNeuronNum);
	int life_idx = getIndex(c_pNetGPU->pNTypes, nTypeNum, LIF);
	int copy_idx = -1;
	real *c_g_vm = NULL;

	if (life_idx >= 0) {
		LIFData *c_g_lif = FROMGPU(static_cast<LIFData *>(c_pNetGPU->ppNeurons[life_idx]), 1);
		c_g_vm = c_g_lif->pV_m;
		copy_idx = life_idx;
	} else {
	}
#endif

	for (int i=0; i<nTypeNum; i++) {
		printf( "Thread %d ntype %d: <<<%d, %d>>>\n", network->_nodeIdx, c_pNetGPU->pNTypes[i], updateSize[c_pNetGPU->pNTypes[i]].gridSize , updateSize[c_pNetGPU->pNTypes[i]].blockSize);
	}
	for (int i=0; i<sTypeNum; i++) {
		printf("Thread %d stype %d: <<<%d. %d>>>\n", network->_nodeIdx, c_pNetGPU->pSTypes[i], updateSize[c_pNetGPU->pSTypes[i]].gridSize, updateSize[c_pNetGPU->pSTypes[i]].blockSize);
	}

	int proc_num = network->_nodeNum;

	vector<int> firedInfo;
	double ts, te;
	ts = MPI_Wtime();

#ifdef PROF
	double t1, t2, t3, t4, t5, t6;
	double comp_time = 0, comm_time = 0, sync_time = 0;
	int *send_count = (int *)malloc(proc_num * sizeof(int));
	int *recv_count = (int *)malloc(proc_num * sizeof(int));
	memset(send_count, 0, proc_num * sizeof(int));
	memset(recv_count, 0, proc_num * sizeof(int));
#endif

	print_gmem("After build");

#ifdef LOG_DATA
	cm->log((string("proc_") + std::to_string(network->_nodeIdx)).c_str());
#endif

	// to_attach();
	printf("Start runing for %d cycles\n", network->_simCycle);

	for (int time=0; time<network->_simCycle; time++) {
#ifdef PROF
		t1 = MPI_Wtime();
#endif
		update_time<<<1, 1>>>(g_buffer->_fired_sizes, max_delay, time);

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppNeurons[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
		}

#ifdef PROF
		t2 = MPI_Wtime();
		comp_time += t2-t1;
#endif
	    cs[thread_id]->fetch_gpu(cm, g_buffer->_fire_table, g_buffer->_fired_sizes, g_buffer->_fire_table_cap, proc_num, max_delay, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * proc_num, c_g_fired_n_num, sizeof(int)*proc_num, cudaMemcpyDeviceToHost));
		cs[thread_id]->update_gpu(time);

#ifdef PROF
		int curr_delay = time % cs[thread_id]->_min_delay;
		t3 = MPI_Wtime();
		comm_time += t3-t2;

		pthread_barrier_wait(&g_proc_barrier);
	    if (thread_id == 0) {	
			MPI_Barrier(MPI_COMM_WORLD);
		}
		if (curr_delay >= min_delay-1) {
			for (int i=0; i<proc_num; i++) {
				send_count[i] += cs[thread_id]->_send_num[i];
				recv_count[i] += cs[thread_id]->_recv_num[i];
			}
		}
		t6 = MPI_Wtime();
		sync_time += t6- t3;
#endif

#ifdef LOG_DATA
		int currentIdx = time%(max_delay+1);

		if (copy_idx >= 0 && (c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]) > 0) {
			COPYFROMGPU(c_vm, c_g_vm, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		}
#endif

		for (int i=0; i<sTypeNum; i++) {
			assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
		}

#ifdef PROF
		cudaDeviceSynchronize();
		t4 = MPI_Wtime();
		comp_time += t4 - t6;
#endif

#ifdef LOG_DATA
		cs[thread_id]->log_gpu(time, (string("proc_") + std::to_string(network->_nodeIdx)).c_str());
#endif
		cs[thread_id]->upload_gpu(g_buffer->_fire_table, g_buffer->_fired_sizes, buffer._fired_sizes, g_buffer->_fire_table_cap, max_delay, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

#ifdef PROF
		cudaDeviceSynchronize();
		t5 = MPI_Wtime();
		comm_time += t5 - t4;
#endif

#ifdef LOG_DATA
		uinteger_t copySize = 0;
		COPYFROMGPU(&copySize, g_buffer->_fired_sizes + currentIdx, 1);
		assert(copySize <= allNeuronNum);
		if (copySize > 0) {
			COPYFROMGPU(buffer._fire_table, g_buffer->_fire_table + (allNeuronNum*currentIdx), copySize);
		}

		for (int i=0; i<copySize; i++) {
			fprintf(sim_file, "%d ", buffer._fire_table[i]);
		}
		fprintf(sim_file, "\n");

		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf ", c_vm[i]);
		}
		fprintf(v_file, "\n");
#endif
	}
	te = MPI_Wtime();
	printf("Thread %d Simulation finesed in %lfs\n", network->_nodeIdx, te-ts);
#ifdef PROF
	printf("Thread %d Simulation perf %lf:%lf:%lf\n", network->_nodeIdx, comp_time, comm_time, sync_time);

	string send;
	string recv;
	for (int i=0; i<proc_num; i++) {
		send += std::to_string(send_count[i]);
		send += ' ';
		recv += std::to_string(recv_count[i]);
		recv += ' ';
	}

	printf("Thread %d Data Send:%s\n", network->_nodeIdx, send.c_str());
	printf("Thread %d Data Recv:%s\n", network->_nodeIdx, recv.c_str());

	printf("Comm stat: cpu_wait_gpu %lf; gpu_wait %lf; cpu_comm %lf; gpu_comm %lf\n", cs[thread_id]->_cpu_wait_gpu, cs[thread_id]->_gpu_wait, cs[thread_id]->_cpu_time, cs[thread_id]->_gpu_time);
#endif

	char name[512];
	sprintf(name, "gpu_mpi_%d", network->_nodeIdx); 

	for (int i=0; i<nTypeNum; i++) {
		cudaLogRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], c_pNetGPU->ppNeurons[i],  name);
	}

	fclose(sim_file);
	fclose(v_file);

	// free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);

	// print_mem("After Free");

	return 0;
}

