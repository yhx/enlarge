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
#include "../utils/FileOp.h"
#include "../utils/TypeFunc.h"
#include "../gpu_utils/mem_op.h"
// #include "../gpu_utils/gpu_utils.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/GBuffers.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "MultiNodeSimulator.h"
// #include "../gpu_utils/gpu_func.h"

#define ASYNC

using std::cout;
using std::endl;

int run_node_cpu(DistriNetwork *network, CrossNodeData *cnd);
int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd);

MultiNodeSimulator::MultiNodeSimulator(Network *network, real dt) : Simulator(network, dt)
{
}

MultiNodeSimulator::~MultiNodeSimulator()
{
}

int MultiNodeSimulator::mpi_init(int *argc, char ***argv)
{
	MPI_Init(argc, argv);
	return 0;
}

int MultiNodeSimulator::run(real time, bool gpu)
{
	FireInfo log;
	run(time, log, gpu);
	return 0;
}

int MultiNodeSimulator::run(real time, FireInfo &log)
{
	run(time, log, true);
	return 0;
}

int MultiNodeSimulator::run(real time, FireInfo &log, bool gpu)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &node_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, node_id, node_num);

	int sim_cycle = round(time/_dt);
	reset();

	SimInfo info(_dt);

	DistriNetwork *network = NULL;
	CrossNodeData *data = NULL;

	if (node_id == 0) {
#if 1
		_network->setNodeNum(node_num);
		DistriNetwork *node_nets = _network->buildNetworks(info);
		CrossNodeData *node_datas = _network->arrangeCrossNodeData(node_num);

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
#ifdef DEBUG
			printf("Send to %d, tag: %d\n", i, DATA_TAG);
#endif
			sendDistriNet(&(node_nets[i]), i, DATA_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("Send DistriNet to %d, tag: %d\n", i, DATA_TAG);
#endif
			sendCND(&(node_datas[i]), i, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("Send CND to %d, tag: %d\n", i, DATA_TAG);
#endif
		}
#else
		network = initDistriNet(1, _dt);
		network->_network = _network->buildNetwork(info);
		network->_simCycle = sim_cycle;
		network->_nodeIdx = 0;
		network->_nodeNum = node_num;
		network->_dt = _dt;
		data = NULL;
#endif
	} else {
#if 1
#ifdef DEBUG
			printf("%d recv from %d, tag: %d\n", node_id, 0, DATA_TAG);
#endif
		network = recvDistriNet(0, DATA_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("%d recv DistriNet from %d, tag: %d\n", node_id, 0, DATA_TAG);
#endif
		data = recvCND(0, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("%d recv CND from %d, tag: %d\n", node_id, 0, DATA_TAG);
#endif
#endif
	}
#ifdef LOG_DATA
	char filename[512];
	sprintf(filename, "net_%d.save", network->_nodeIdx); 
	FILE *net_file = openFile(filename, "w+");
	saveDistriNet(network, net_file);
#endif

	MPI_Barrier(MPI_COMM_WORLD);


	if (gpu) {
		run_node_gpu(network, data);
	} else {
		run_node_cpu(network, data);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}


int run_node_cpu(DistriNetwork *network, CrossNodeData *cnd) {
	char log_filename[512];
	sprintf(log_filename, "sim.mpi_%d.log", network->_nodeIdx); 
	FILE *log_file = fopen(log_filename, "w+");
	assert(log_file != NULL);

	char v_filename[512];
	sprintf(v_filename, "v.mpi_%d.log", network->_nodeIdx); 
	FILE *v_file = fopen(v_filename, "w+");
	assert(v_file != NULL);

	GNetwork *pNetCPU = network->_network;

	int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
	int nodeNeuronNum = pNetCPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->pConnection->nNum;
	int nodeSynapseNum = pNetCPU->pSynapseNums[sTypeNum];
	printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d NeuronNum: %d, AllNeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, allNeuronNum, nodeSynapseNum);

	int maxDelay = pNetCPU->pConnection->maxDelay;
	int minDelay = pNetCPU->pConnection->minDelay;

	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	int cFiredTableCap = allNeuronNum;


	real *c_gNeuronInput = (real*)malloc(sizeof(real)*allNeuronNum);
	memset(c_gNeuronInput, 0, sizeof(real)*allNeuronNum);
	real *c_gNeuronInput_I = (real*)malloc(sizeof(real)*allNeuronNum); 
	memset(c_gNeuronInput_I, 0, sizeof(real)*allNeuronNum);
	int *c_gFiredTable = (int*)malloc(sizeof(int)*allNeuronNum*(maxDelay+1));
	memset(c_gFiredTable, 0, sizeof(int)*allNeuronNum*(maxDelay+1));
   	int *c_gFiredTableSizes = (int*)malloc(sizeof(int)*(maxDelay+1));
   	memset(c_gFiredTableSizes, 0, sizeof(int)*(maxDelay+1));

#ifdef LOG_DATA
	int copy_idx = getIndex(pNetCPU->pNTypes, nTypeNum, LIF);
#endif

	printf("Start runing for %d cycles\n", network->_simCycle);
	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);

	for (int time=0; time<network->_simCycle; time++) {
		int currentIdx = time % (maxDelay+1);
		c_gFiredTableSizes[currentIdx] = 0;

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->pConnection, pNetCPU->ppNeurons[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

#if 1
		memset(cnd->_send_num, 0, sizeof(int)*(cnd->_node_num));
		generateCND(pNetCPU->pConnection, c_gFiredTable, c_gFiredTableSizes, network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, cnd->_send_data, cnd->_send_offset, cnd->_send_num, network->_nodeNum, time, cFiredTableCap);

		MPI_Alltoall(cnd->_send_num, 1, MPI_INT, cnd->_recv_num, 1,MPI_INT,MPI_COMM_WORLD);

		for (int i=0; i<cnd->_node_num; i++) {
			cnd->_recv_offset[i+1] = cnd->_recv_offset[i] + cnd->_recv_num[i];
		}

#ifdef ASYNC
		MPI_Request request_t;
		MPI_Status status_t;
		int ret = MPI_Ialltoallv(cnd->_send_data, cnd->_send_num, cnd->_send_offset , MPI_INT, 
				cnd->_recv_data, cnd->_recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD, &request_t);
		assert(ret == MPI_SUCCESS);
#else
		int ret = MPI_Alltoallv(cnd->_send_data, cnd->_send_num, cnd->_send_offset, MPI_INT, 
				cnd->_recv_data, cnd->_recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
#endif
#endif



		for (int i=0; i<sTypeNum; i++) {
			updateType[pNetCPU->pSTypes[i]](pNetCPU->pConnection, pNetCPU->ppSynapses[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time);
		}

#if 1
#ifdef ASYNC
		ret = MPI_Wait(&request_t, &status_t);
		assert(ret == MPI_SUCCESS);
#endif

		int delay_idx = time % (maxDelay + 1);

		for (int i=0; i<cnd->_recv_offset[cnd->_node_num]; i++) {
			c_gFiredTable[allNeuronNum*delay_idx + c_gFiredTableSizes[delay_idx] + i] = cnd->_recv_data[i];
		}
		c_gFiredTableSizes[delay_idx] += cnd->_recv_offset[cnd->_node_num];
#endif

#ifdef LOG_DATA
		LIFData *c_lif = (LIFData *)pNetCPU->ppNeurons[copy_idx];
		real *c_vm = c_lif->pV_m;


		for (int i=0; i<pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(v_file, "\n");
		int copySize = c_gFiredTableSizes[currentIdx];

		for (int i=0; i<copySize; i++) {
			fprintf(log_file, "%d ", c_gFiredTable[allNeuronNum*currentIdx+i]);
		}
		fprintf(log_file, "\n");

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

	fclose(log_file);
	fclose(v_file);

	return 0;
}

int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd) {
	char log_filename[512];
	sprintf(log_filename, "sim.gpu.mpi_%d.log", network->_nodeIdx); 
	FILE *log_file = fopen(log_filename, "w+");
	assert(log_file != NULL);

	char v_filename[512];
	sprintf(v_filename, "v.gpu.mpi_%d.log", network->_nodeIdx); 
	FILE *v_file = fopen(v_filename, "w+");
	assert(v_file != NULL);

	checkCudaErrors(cudaSetDevice(0));

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
		cudaGenerateCND<<<(allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_g_idx2index, c_g_cross_index2idx, cnd_gpu->_send_data, cnd_gpu->_send_offset, cnd_gpu->_send_num, network->_nodeNum, time);

		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * network->_nodeNum, c_g_fired_n_num, sizeof(int)*network->_nodeNum, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(cnd->_send_num, cnd_gpu->_send_num, sizeof(int)*(cnd->_node_num), cudaMemcpyDeviceToHost));
		for (int i=0; i< network->_nodeNum; i++) {
			if (cnd->_send_num[i] > 0) {
				checkCudaErrors(cudaMemcpy((&cnd->_send_data[cnd->_send_offset[i]]), &(cnd_gpu->_send_data[cnd->_send_offset[i]]), sizeof(int)*(cnd->_send_num[i]), cudaMemcpyDeviceToHost));
			}
		}

		int ret = MPI_Alltoall(cnd->_send_num, 1, MPI_INT, cnd->_recv_num, 1,MPI_INT,MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);

		for (int i=0; i<cnd->_node_num; i++) {
			cnd->_recv_offset[i+1] = cnd->_recv_offset[i] + cnd->_recv_num[i];
		}

		MPI_Request request_t;
		MPI_Status status_t;
		ret = MPI_Ialltoallv(cnd->_send_data, cnd->_send_num, cnd->_send_offset , MPI_INT, 
				cnd->_recv_data, cnd->_recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD, &request_t);
		assert(ret == MPI_SUCCESS);

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

		ret = MPI_Wait(&request_t, &status_t);
		assert(ret == MPI_SUCCESS);

		int delay_idx = time % (maxDelay + 1);

		int firedSize = 0;
		checkCudaErrors(cudaMemcpy(&firedSize, buffers->c_gFiredTableSizes+delay_idx, sizeof(int), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(buffers->c_gFiredTable + allNeuronNum * delay_idx + firedSize , cnd->_recv_data, sizeof(int)*(cnd->_recv_offset[cnd->_node_num]), cudaMemcpyHostToDevice));
		cudaUpdateFTS<<<1, 1>>>(buffers->c_gFiredTableSizes, cnd->_recv_offset[cnd->_node_num], delay_idx);

		// for (int i=0; i< network->_nodeNum; i++) {
			// int i2idx = network->_nodeIdx + network->_nodeNum * i;
			// if (gCrossDataGPU->_firedNum[i2idx] > 0) {
			// 	int num = gCrossDataGPU->_firedNum[i2idx];
			// 	cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
			// }
		//}
		
		
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

