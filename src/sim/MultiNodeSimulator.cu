
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
#include "../neuron/iaf/IAFData.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/helper/helper_gpu.h"
#include "../../msg_utils/msg_utils/msg_utils.h"
#include "../../msg_utils/msg_utils/GPUManager.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/gpu_utils.h"
#include "MultiNodeSimulator.h"
// #include "../gpu_utils/GBuffers.h"


int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd, int gpu) 
{
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

	// print_mem("Before SetDevice");
	gm.set(network->_nodeIdx % gpu);
	gm.lock();
	// print_mem("Before Network");

	GNetwork *pNetCPU = network->_network;
	print_gmem("Before using GPU");
	std::cout << "BBefore using GPU" << std::endl;
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);
	// print_mem("Copied Network");

	CrossNodeData * cnd_gpu = copyCNDtoGPU(cnd);
	// print_mem("Copied CND");

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
	int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("Thread %d, NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d, NodeNeuronNum: %d, AllNeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, allNeuronNum, nodeSynapseNum);

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	int minDelay = pNetCPU->ppConnections[0]->minDelay;

	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	// print_mem("Before Buffers");

	// GBuffers *buffers = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->ppConnections[0]->maxDelay, network->_dt);

	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, maxDelay, 0);
	Buffer *g_buffer = buffer._gpu_array;

	int *c_fired_sizes = hostMalloc<int>(maxDelay+1);

	BlockSize *updateSize = getBlockSize(allNeuronNum, nodeSynapseNum);

	// print_mem("Alloced Buffers");

#ifdef LOG_DATA
	real *c_vm = hostMalloc<real>(nodeNeuronNum);
	int life_idx = getIndex(c_pNetGPU->pNTypes, nTypeNum, LIF);
	int copy_idx = -1;
	real *c_g_vm = NULL;
	// std::cout << "life_idx: " << life_idx << std::endl;
	if (life_idx >= 0) {  // 如果有当前类型的神经元
		LIFData *c_g_lif = FROMGPU(static_cast<LIFData *>(c_pNetGPU->ppNeurons[life_idx]), 1);
		c_g_vm = c_g_lif->pV_m;
		copy_idx = life_idx;
		// for (size_t i = 0; i < nTypeNum; ++i) {
		// 	c_g_vm = cudaGetVNeuron[pNetCPU->pNTypes[i]](c_pNetGPU->ppNeurons[i]);
		// 	copy_idx = i;
		// }
	} else {  // 如果没有当前类型的神经元，即为IAF类型的神经元 
		// TODO: 将其改为函数指针的实现方式
		life_idx = getIndex(c_pNetGPU->pNTypes, nTypeNum, IAF);
		IAFData *c_g_iaf = FROMGPU(static_cast<IAFData *>(c_pNetGPU->ppNeurons[life_idx]), 1);
		c_g_vm = c_g_iaf->pV_m;
		copy_idx = life_idx;
	}

#endif

	for (int i=0; i<nTypeNum; i++) {
		printf( "Thread %d ntype %d: <<<%d, %d>>>\n", network->_nodeIdx, c_pNetGPU->pNTypes[i], updateSize[c_pNetGPU->pNTypes[i]].gridSize , updateSize[c_pNetGPU->pNTypes[i]].blockSize);
	}
	for (int i=0; i<sTypeNum; i++) {
		printf("Thread %d stype %d: <<<%d. %d>>>\n", network->_nodeIdx, c_pNetGPU->pSTypes[i], updateSize[c_pNetGPU->pSTypes[i]].gridSize, updateSize[c_pNetGPU->pSTypes[i]].blockSize);
	}

	int node_num = network->_nodeNum;

	CrossNodeMap *cnm_gpu = to_gpu(network->_crossnodeMap);

	// integer_t * c_g_idx2index = copyToGPU(network->_crossnodeMap->_idx2index, allNeuronNum);
	// integer_t * c_g_cross_index2idx = copyToGPU(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossSize);
	// int * c_g_global_cross_data = gpuMalloc<int>(allNeuronNum * node_num);
	// int * c_g_fired_n_num = gpuMalloc<int>(node_num);

	vector<int> firedInfo;
	// double ts, te;
	// ts = MPI_Wtime();

#ifdef PROF
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, tss = 0, tee = 0;
	double comp_time = 0, comm_time = 0, sync_time = 0, comm_time2 = 0, gpu_time = 0, gpu_wait = 0;
	double *t_neuron, *t_synapse;
	t_neuron = malloc_c<double>(pNetCPU->nTypeNum);
	t_synapse = malloc_c<double>(pNetCPU->sTypeNum);
	int *send_count = (int *)malloc(node_num * sizeof(int));
	int *recv_count = (int *)malloc(node_num * sizeof(int));
	memset(send_count, 0, node_num * sizeof(int));
	memset(recv_count, 0, node_num * sizeof(int));
#endif

	print_gmem("After build");

	to_attach();
	printf("Start runing for %d cycles\n", network->_simCycle);
	
	MPI_Barrier(MPI_COMM_WORLD);
	double ts, te;
	ts = MPI_Wtime();

	for (int time=0; time<network->_simCycle; time++) {
#ifdef PROF
		if (network->_nodeIdx == 0 && time % 100 == 1) {
			printf("%d\n", time);
		}
		t1 = MPI_Wtime();
#endif
		update_time<<<1, 1>>>(g_buffer->_fired_sizes, maxDelay, time);

#ifdef LOG_DATA
		COPYFROMGPU(buffer._data, g_buffer->_data, buffer._data_size);
#endif

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
#ifdef PROF
			tss = MPI_Wtime();
#endif
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppNeurons[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
#ifdef PROF
			cudaDeviceSynchronize();
			tee = MPI_Wtime();
			t_neuron[i] += tee - tss;
#endif
		}

#ifdef PROF
		// cudaDeviceSynchronize();
		t2 = MPI_Wtime();
		comp_time += t2-t1;
#endif
		int curr_delay = time % cnd->_min_delay;
		cudaGenerateCND(cnm_gpu->_idx2index, cnm_gpu->_crossnodeIndex2idx, cnd_gpu, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, maxDelay, cnd_gpu->_min_delay, node_num, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * node_num, c_g_fired_n_num, sizeof(int)*node_num, cudaMemcpyDeviceToHost));
		MPI_Request request_t;
		update_cnd_gpu(cnd_gpu, cnd, curr_delay, &request_t);

#ifdef PROF
		// cudaDeviceSynchronize();
		t3 = MPI_Wtime();
		comm_time += t3-t2;
		MPI_Barrier(MPI_COMM_WORLD);
		if (curr_delay >= minDelay-1) {
			for (int i=0; i<node_num; i++) {
				send_count[i] += cnd->_send_num[i];
				recv_count[i] += cnd->_recv_num[i];
			}
		}
		t6 = MPI_Wtime();
		sync_time += t6- t3;
#endif

#ifdef LOG_DATA
		int currentIdx = time%(maxDelay+1);

		uinteger_t copySize = 0;
		COPYFROMGPU(&copySize, g_buffer->_fired_sizes + currentIdx, 1);
		assert(copySize <= allNeuronNum);
		if (copySize > 0) {
			COPYFROMGPU(buffer._fire_table, g_buffer->_fire_table + (allNeuronNum*currentIdx), copySize);
		}

		if (copy_idx >= 0 && (c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]) > 0) {
			COPYFROMGPU(c_vm, c_g_vm, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		}
#endif

		for (int i=0; i<sTypeNum; i++) {
			assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
#ifdef PROF
			tss = MPI_Wtime();
#endif
			cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
#ifdef PROF
			cudaDeviceSynchronize();
			tee = MPI_Wtime();
			t_synapse[i] += tee - tss;
#endif
		}

#ifdef PROF
		// cudaDeviceSynchronize();
		t4 = MPI_Wtime();
		comp_time += t4 - t6;
#endif
		if (curr_delay >= minDelay -1) {
#ifdef PROF
			double t_ts = 0, t_te = 0;
#endif
#ifdef PROF
			t_ts = MPI_Wtime();
#endif
			MPI_Status status_t;
			int ret = MPI_Wait(&request_t, &status_t);
			assert(ret == MPI_SUCCESS);
#ifdef PROF
			t_te = MPI_Wtime();
			comm_time2 += t_te - t_ts;
#endif
#ifdef PROF
			t_ts = MPI_Wtime();
#endif
			checkCudaErrors(cudaMemcpy(c_fired_sizes, g_buffer->_fired_sizes, sizeof(int)*(maxDelay+1), cudaMemcpyDeviceToHost));
#ifdef PROF
			t_te = MPI_Wtime();
			gpu_wait += t_te - t_ts;
#endif

#ifdef PROF
			t_ts = MPI_Wtime();
#endif 
			for (int d_=0; d_ < minDelay; d_++) {
				int delay_idx = (time-minDelay+2+d_+maxDelay)%(maxDelay+1);
				for (int n_ = 0; n_<node_num; n_++) {
					int start = cnd->_recv_start[n_*(minDelay+1)+d_];
					int end = cnd->_recv_start[n_*(minDelay+1)+d_+1];
					if (end > start) {
						assert(c_fired_sizes[delay_idx] + end - start <= allNeuronNum);
						checkCudaErrors(cudaMemcpy(g_buffer->_fire_table + allNeuronNum*delay_idx + c_fired_sizes[delay_idx], cnd->_recv_data + cnd->_recv_offset[n_] + start, sizeof(int)*(end-start), cudaMemcpyHostToDevice));
						c_fired_sizes[delay_idx] += end - start;
					}
				}
			}
			checkCudaErrors(cudaMemcpy(g_buffer->_fired_sizes, c_fired_sizes, sizeof(int)*(maxDelay+1), cudaMemcpyHostToDevice));
#ifdef PROF
			t_te = MPI_Wtime();
			gpu_time += t_te - t_ts;
#endif
#ifdef LOG_DATA
			log_cnd(cnd, time, send_file, recv_file);
#endif
			reset_cnd_gpu(cnd_gpu, cnd);
		}


#ifdef PROF
		// cudaDeviceSynchronize();
		t5 = MPI_Wtime();
		comm_time += t5 - t4;
#endif

		// for (int i=0; i< node_num; i++) {
			// int i2idx = network->_nodeIdx + node_num * i;
			// if (gCrossDataGPU->_firedNum[i2idx] > 0) {
			// 	int num = gCrossDataGPU->_firedNum[i2idx];
			// 	cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->ppConnections[0], g_buffer->_fire_table, g_buffer->_fired_sizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
			// }
		//}
		
		
#ifdef LOG_DATA
		for (int i=0; i<copySize; i++) {
			fprintf(sim_file, "%d ", buffer._fire_table[i]);
		}
		fprintf(sim_file, "\n");

		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf ", c_vm[i]);
		}
		fprintf(v_file, "\n");

		log_array(input_e_file, buffer._data, buffer._data_size);
#endif
	}
	te = MPI_Wtime();
	printf("Thread %d Simulation finesed in %lfs\n", network->_nodeIdx, te-ts);
#ifdef PROF
	printf("Thread %d Simulation perf %lf:%lf:%lf\n", network->_nodeIdx, comp_time, comm_time, sync_time);

	printf("Prof: %lf:%lf:%lf:%lf:%lf\n", cnd_gpu->_cpu_wait_gpu, cnd->_cpu_time, comm_time2, gpu_time, gpu_wait);

	
	string send;
	string recv;
	for (int i=0; i<node_num; i++) {
		send += std::to_string(send_count[i]);
		send += ' ';
		recv += std::to_string(recv_count[i]);
		recv += ' ';
	}

	printf("Thread %d Data Send:%s\n", network->_nodeIdx, send.c_str());
	printf("Thread %d Data Recv:%s\n", network->_nodeIdx, recv.c_str());
	if (network->_nodeIdx == 0) {
		printf("neuron time: \n");
		for (int i = 0; i < pNetCPU->nTypeNum; ++i) {
			printf("%d %lf ", pNetCPU->pNTypes[i], t_neuron[i]);
		}
		printf("\n");
		printf("synapse time: \n");
		for (int i = 0; i < pNetCPU->sTypeNum; ++i) {
			printf("%d %lf ", pNetCPU->pSTypes[i], t_synapse[i]);
		}
		printf("\n");
	}
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

