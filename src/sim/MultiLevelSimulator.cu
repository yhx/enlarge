
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
#include "../../msg_utils/msg_utils/ProcBuf.h"
#include "../../msg_utils/msg_utils/GPUManager.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/gpu_utils.h"
#include "MultiLevelSimulator.h"
// #include "../gpu_utils/GBuffers.h"

pthread_barrier_t g_proc_barrier;

void * run_gpu_ml(void *para) {
	RunPara * tmp = static_cast<RunPara*>(para);
	DistriNetwork *network = tmp->_net;
	CrossMap *cm = tmp->_cm;
	ProcBuf *pbuf = tmp->_pbuf;
	int thread_id = tmp->_thread_id;
	// print_mem("Inside Run");

#ifdef LOG_DATA
	FILE *v_file = log_file_mpi("v", network->_nodeIdx);
	FILE *sim_file = log_file_mpi("sim", network->_nodeIdx);
	FILE *msg_file = log_file_mpi("msg", network->_nodeIdx);
	FILE *send_file = log_file_mpi("send", network->_nodeIdx);
	FILE *recv_file = log_file_mpi("recv", network->_nodeIdx);
	FILE *input_i_file = log_file_mpi("input_i", network->_nodeIdx);
	FILE *input_e_file = log_file_mpi("input_e", network->_nodeIdx);
#endif

	gm.set(thread_id);
	gm.lock();
    pbuf->create_cuda_stream(thread_id);

	GNetwork *pNetCPU = network->_network;
	print_gmem("Before using GPU");
/*
#ifdef PROF
	double bef_gpu = print_gmem("Before using GPU");
#endif
*/
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);
	// print_mem("Copied Network");

	cm->to_gpu();
	pbuf->to_gpu(thread_id);
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

	int proc_num = network->_nodeNum;

	vector<int> firedInfo;
	double ts, te;
	

#ifdef PROF
	double t1, t2, t3, t4, t5, t6, tss=0, tee=0;
	double comp_time = 0, comm_time = 0, sync_time = 0;
	double *t_neuron, *t_synapse;
	t_neuron = malloc_c<double>(pNetCPU->nTypeNum);
	t_synapse = malloc_c<double>(pNetCPU->sTypeNum);
	long long *send_count = malloc_c<long long>(proc_num);
	long long *recv_count = malloc_c<long long>(proc_num);
	memset(send_count, 0, proc_num * sizeof(int));
	memset(recv_count, 0, proc_num * sizeof(int));
#endif

	print_gmem("After build");
/*
#ifdef PROF
	double aft_gpu = print_gmem("After build");
	printf("[FINAL] After build used [%lf] MB GPU MEM!\n", aft_gpu - bef_gpu);
#endif
*/

#ifdef LOG_DATA
	cm->log((string("proc_") + std::to_string(network->_nodeIdx)).c_str());
#endif

	// to_attach();
	printf("Start runing for %d cycles\n", network->_simCycle);
    ts = MPI_Wtime();
	for (int time=0; time<network->_simCycle; time++) {
		// printf("%d\n", time);
		// printf("Rank %d: %d\n", network->_nodeIdx, time);
#ifdef PROF
		// printf("Rank %d: %d\n", network->_nodeIdx, time);
		// if (thread_id == 0 && time % 100 == 0) {
		// 	printf("%d\n", time);
		// }
		t1 = MPI_Wtime();
#endif

#ifdef PROF
		t1 = MPI_Wtime();
#endif
		update_time<<<1, 1>>>(g_buffer->_fired_sizes, max_delay, time);
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
        checkCudaErrors(cudaDeviceSynchronize());
#ifdef PROF
		// cudaDeviceSynchronize();
		// printf("Rank %d: after update neuron\n", network->_nodeIdx);
		t2 = MPI_Wtime();
		comp_time += t2-t1;
#endif
	    pbuf->fetch_gpu(thread_id, cm, g_buffer->_fire_table, g_buffer->_fired_sizes, g_buffer->_fire_table_cap, max_delay, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
        
        // COPYFROMGPU(buffer._fired_sizes, g_buffer->_fired_sizes, max_delay + 1);  // 直接从GPU中拷数据到CPU中
		pbuf->update_gpu_ml2(thread_id, time, g_buffer->_fire_table, g_buffer->_fired_sizes, buffer._fired_sizes, g_buffer->_fire_table_cap, max_delay);
        // pbuf->update_gpu_ml(thread_id, time);
        
#ifdef PROF
		int curr_delay = time % pbuf->_min_delay;
		t3 = MPI_Wtime();
		comm_time += t3-t2;
		pthread_barrier_wait(&g_proc_barrier);
	    if (thread_id == 0) {	
			MPI_Barrier(MPI_COMM_WORLD);
		}
		if (curr_delay >= min_delay-1 && thread_id == 0) {
			for (int i=0; i<pbuf->_proc_num; i++) {
				send_count[i] += pbuf->_send_num[i];
				recv_count[i] += pbuf->_recv_num[i];
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
        // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id + pbuf->_thread_num]));
        // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id]));
        // checkCudaErrors(cudaDeviceSynchronize());
        // pthread_barrier_wait(pbuf->_barrier);
		for (int i=0; i<sTypeNum; i++) {
			assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
#ifdef PROF
			tss = MPI_Wtime();
#endif
			cudaUpdateSynapse[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]], pbuf->_cuda_stream[thread_id + pbuf->_thread_num]);
            // cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], g_buffer->_data, g_buffer->_fire_table, g_buffer->_fired_sizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
#ifdef PROF
			cudaDeviceSynchronize();
			tee = MPI_Wtime();
			t_synapse[i] += tee - tss;
#endif
		}

#ifdef PROF
		// cudaDeviceSynchronize();
		// printf("Rank %d: after update synapse\n", network->_nodeIdx);
		t4 = MPI_Wtime();
		comp_time += t4 - t6;
#endif

#ifdef LOG_DATA
		// cs[thread_id]->log_gpu(time, (string("proc_") + std::to_string(network->_nodeIdx)).c_str());
		pbuf->log_cpu(thread_id, time, "ml");
#endif
        // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id + pbuf->_thread_num]));
        // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id]));
        // checkCudaErrors(cudaDeviceSynchronize());
        // pthread_barrier_wait(pbuf->_barrier);
        // cudaStreamSynchronize(pbuf->_cuda_stream[thread_id + pbuf->_thread_num]);
		pbuf->upload_gpu_ml(thread_id, g_buffer->_fire_table, g_buffer->_fired_sizes, buffer._fired_sizes, g_buffer->_fire_table_cap, max_delay, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

#ifdef PROF
		// cudaDeviceSynchronize();
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
		fflush(sim_file);

		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf ", c_vm[i]);
		}
		fprintf(v_file, "\n");
		fflush(v_file);
#endif
	}
    // if (thread_id == 0) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // pthread_barrier_wait(pbuf->_barrier);
	te = MPI_Wtime();
	printf("Thread %d Simulation finesed in %lfs\n", network->_nodeIdx, te-ts);
#ifdef PROF
    if (thread_id == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    pthread_barrier_wait(pbuf->_barrier);
	printf("Thread %d Simulation perf %lf:%lf:%lf\n", network->_nodeIdx, comp_time, comm_time, sync_time);

	pbuf->prof();

	string send;
	string recv;
    if (thread_id == 0) {
        for (int i = 0; i < pbuf->_proc_num; i++) {
            send += std::to_string(send_count[i]);
            send += ' ';
            recv += std::to_string(recv_count[i]);
            recv += ' ';
        }
    }
    if (thread_id == 0) {
	    printf("Thread %d Data Send:%s\n", pbuf->_proc_rank, send.c_str());
	    printf("Thread %d Data Recv:%s\n", pbuf->_proc_rank, recv.c_str());
    }
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

	printf("Comm stat %d: cpu_wait_gpu %lf; gpu_wait %lf; cpu_comm %lf; gpu_comm %lf; update_gpu_ml_dirct %lf; _mpi_dataoffset %lf; updata %lf; upload %lf\n", network->_nodeIdx, pbuf->_cs[thread_id]->_cpu_wait_gpu, pbuf->_cs[thread_id]->_gpu_wait, pbuf->_cs[thread_id]->_cpu_time, pbuf->_cs[thread_id]->_gpu_time, pbuf->_cs[thread_id]->_update_gpu_ml_dirct, pbuf->_cs[thread_id]->_mpi_dataoffset, pbuf->_cs[thread_id]->_update_time, pbuf->_cs[thread_id]->_upload_time);
#endif

	char name[512];
	sprintf(name, "gpu_mpi_%d", network->_nodeIdx); 
    // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id + pbuf->_thread_num]));
    // checkCudaErrors(cudaStreamSynchronize(pbuf->_cuda_stream[thread_id]));
    checkCudaErrors(cudaDeviceSynchronize());
	for (int i=0; i<nTypeNum; i++) {
		cudaLogRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], c_pNetGPU->ppNeurons[i], name);
	}

#ifdef LOG_DATA
	fclose(sim_file);
	fclose(v_file);
#endif

	// free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);

	// print_mem("After Free");

	return 0;
}

