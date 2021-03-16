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
#include "../utils/helper_c.h"
#include "../base/TypeFunc.h"
#include "../gpu_utils/helper_gpu.h"
#include "../gpu_utils/runtime.h"
#include "../gpu_utils/GBuffers.h"
#include "../msg_utils/msg_utils.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "MultiNodeSimulator.h"
// #include "../gpu_utils/gpu_func.h"

MultiNodeSimulator::MultiNodeSimulator(Network *network, real dt) : Simulator(network, dt)
{
	_node_nets = NULL;
	_node_datas = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &_node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &_node_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, _node_id, _node_num);
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

int MultiNodeSimulator::build_net()
{

	SimInfo info(_dt);
	info.save_mem = true;

	if (!_node_nets) {
		_network->set_node_num(_node_num);
		_node_nets = _network->buildNetworks(info);
		for (int i=0; i<_node_num; i++) {
			_node_nets[i]._simCycle = 0;
			_node_nets[i]._nodeIdx = i;
			_node_nets[i]._nodeNum = _node_num;
			_node_nets[i]._dt = _dt;
		}
	}

	if (!_node_datas) {
		_node_datas = _network->arrangeCrossNodeData(_node_num, info);
	}

	return 0;
}

int MultiNodeSimulator::save_net(const char *name)
{
	if (_node_nets && _node_datas) {
		char name_t[512];
		sprintf(name_t, "%s.num", name);
		FILE *f = openFile(name_t, "w+");
		fwrite_c(&(_node_num), sizeof(int), 1, f);
		closeFile(f);

		for (int i=0; i<_node_num; i++) {
			sprintf(name_t, "%s_%d.net", name, i);
			f = openFile(name_t, "w+");
			saveDistriNet(&(_node_nets[i]), f);
			closeFile(f);
		}

		for (int i=0; i<_node_num; i++) {
			sprintf(name_t, "%s_%d.cnd", name, i);
			f = openFile(name_t, "w+");
			saveCND(&(_node_datas[i]), f);
			closeFile(f);
		}
	} else {
		printf("Before save, build the net first\n");
		return 1;
	}

	return 0;
}

int MultiNodeSimulator::load_net(const char *name)
{
	return 0;
}

int MultiNodeSimulator::distribute(DistriNetwork **pp_net, CrossNodeData **pp_data, SimInfo &info, int sim_cycle)
{

	if (_node_id == 0) {
#if 1
		// print_mem("Finish Network");
		// print_mem("Finish CND");
		if (!(_node_nets && _node_datas)) {
			build_net();
		}

		for (int i=0; i<_node_num; i++) {
			_node_nets[i]._simCycle = sim_cycle;
			_node_nets[i]._nodeIdx = i;
			_node_nets[i]._nodeNum = _node_num;
			_node_nets[i]._dt = _dt;
		}

		*pp_net = &(_node_nets[0]);
		*pp_data = &(_node_datas[0]);
		allocDataCND(*pp_data);
		// print_mem("AllocData CND");

		for (int i=1; i<_node_num; i++) {
#ifdef DEBUG
			printf("Send to %d, tag: %d\n", i, DATA_TAG);
#endif
			sendDistriNet(&(_node_nets[i]), i, DATA_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("Send DistriNet to %d, tag: %d\n", i, DATA_TAG);
#endif
			sendCND(&(_node_datas[i]), i, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
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
		printf("%d recv from %d, tag: %d\n", _node_id, 0, DATA_TAG);
#endif
		*pp_net = recvDistriNet(0, DATA_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
		printf("%d recv DistriNet from %d, tag: %d\n", _node_id, 0, DATA_TAG);
#endif
		*pp_data = recvCND(0, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
#ifdef DEBUG
		printf("%d recv CND from %d, tag: %d\n", _node_id, 0, DATA_TAG);
#endif
#endif
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;
}

int MultiNodeSimulator::run(real time, FireInfo &log, bool gpu)
{

	int sim_cycle = round(time/_dt);
	reset();

	SimInfo info(_dt);
	info.save_mem = true;

	DistriNetwork *network = NULL;
	CrossNodeData *data = NULL;

	to_attach();

	distribute(&network, &data, info, sim_cycle);

#ifdef LOG_DATA
	char filename[512];
	sprintf(filename, "net_%d.save", network->_nodeIdx); 
	FILE *net_file = openFile(filename, "w+");
	saveDistriNet(network, net_file);
#endif


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
	// char sim_filename[512];
	// sprintf(sim_filename, "sim.mpi_%d.log", network->_nodeIdx); 
	// FILE *sim_file = fopen(sim_filename, "w+");
	// assert(sim_file != NULL);

	// char v_filename[512];
	// sprintf(v_filename, "v.mpi_%d.log", network->_nodeIdx); 
	// FILE *v_file = fopen(v_filename, "w+");
	// assert(v_file != NULL);

	// char msg_filename[512];
	// sprintf(msg_filename, "msg.mpi_%d.log", network->_nodeIdx); 
	// FILE *msg_file = fopen(msg_filename, "w+");
	// assert(msg_file != NULL);

	FILE *v_file = log_file_mpi("v", network->_nodeIdx);
	FILE *sim_file = log_file_mpi("sim", network->_nodeIdx);
#ifdef LOG_DATA
	FILE *msg_file = log_file_mpi("msg", network->_nodeIdx);
	FILE *send_file = log_file_mpi("send", network->_nodeIdx);
	FILE *recv_file = log_file_mpi("recv", network->_nodeIdx);
	FILE *input_i_file = log_file_mpi("input_i", network->_nodeIdx);
	FILE *input_e_file = log_file_mpi("input_e", network->_nodeIdx);
#endif

	GNetwork *pNetCPU = network->_network;

	int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
	int nodeNeuronNum = pNetCPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
	int nodeSynapseNum = pNetCPU->pSynapseNums[sTypeNum];
	printf("Thread %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d NeuronNum: %d, AllNeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, allNeuronNum, nodeSynapseNum);

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	int minDelay = pNetCPU->ppConnections[0]->minDelay;
	assert(minDelay == cnd->_min_delay);

	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	int cFiredTableCap = allNeuronNum;

	int node_num = cnd->_node_num;

	real *c_gNeuronInput = (real*)malloc(sizeof(real)*allNeuronNum);
	memset(c_gNeuronInput, 0, sizeof(real)*allNeuronNum);
	real *c_gNeuronInput_I = (real*)malloc(sizeof(real)*allNeuronNum); 
	memset(c_gNeuronInput_I, 0, sizeof(real)*allNeuronNum);
	size_t *c_gFiredTable = (size_t*)malloc(sizeof(size_t)*allNeuronNum*(maxDelay+1));
	memset(c_gFiredTable, 0, sizeof(size_t)*allNeuronNum*(maxDelay+1));
   	size_t *c_gFiredTableSizes = (size_t*)malloc(sizeof(size_t)*(maxDelay+1));
   	memset(c_gFiredTableSizes, 0, sizeof(size_t)*(maxDelay+1));

   	c_gFiredCount = (int*)malloc(sizeof(int)*(allNeuronNum));
   	memset(c_gFiredCount, 0, sizeof(int)*(allNeuronNum));

#ifdef LOG_DATA
	int copy_idx = getIndex(pNetCPU->pNTypes, nTypeNum, LIF);
#endif

	printf("Start runing for %d cycles\n", network->_simCycle);
	vector<int> firedInfo;
	struct timeval ts, te;
	gettimeofday(&ts, NULL);

#ifdef PROF
	struct timeval t1, t2, t3, t4, t5, t6;
	double comp_time = 0, comm_time = 0, sync_time = 0;
#endif
#ifdef DEBUG
	printf("Cycles: ");
#endif 

	for (int time=0; time<network->_simCycle; time++) {
#ifdef PROF
		gettimeofday(&t1, NULL);
#endif
// #ifdef DEBUG
// 		printf("%d ", time);
// 		fflush(stdout);
// #endif 
		int currentIdx = time % (maxDelay+1);
		c_gFiredTableSizes[currentIdx] = 0;

		// if (time == 38) {
		// 	to_attach();
		// }

#ifdef LOG_DATA
		log_array(input_e_file, c_gNeuronInput, nodeNeuronNum);
		log_array(input_i_file, c_gNeuronInput_I, nodeNeuronNum);
#endif

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

#if 1
		// memset(cnd->_send_num, 0, sizeof(int)*(node_num));
#ifdef PROF
		gettimeofday(&t2, NULL);
		comp_time += 1000000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec);
#endif
		int curr_delay = time % cnd->_min_delay;
		generateCND(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, cnd, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, maxDelay, cnd->_min_delay, network->_nodeNum, time);


		MPI_Request request_t;
		update_cnd(cnd, curr_delay, &request_t);
		// if (curr_delay >= minDelay - 1) {
		// 	msg_cnd(cnd, &request_t);
		// } else {
		// 	for (int i=0; i<node_num; i++) {
		// 		cnd->_send_start[i*(minDelay+1)+curr_delay+2] = cnd->_send_num[i*(minDelay+1)+curr_delay+1];
		// 	}
		// }
#endif
#ifdef PROF
		gettimeofday(&t3, NULL);
		comm_time += 1000000 * (t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec);
#endif
#ifdef PROF

		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&t6, NULL);
		sync_time += 1000000 * (t6.tv_sec - t3.tv_sec) + (t6.tv_usec - t3.tv_usec);
#endif

		for (int i=0; i<sTypeNum; i++) {
			updateType[pNetCPU->pSTypes[i]](pNetCPU->pConnection, pNetCPU->ppSynapses[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time);
		}

#ifdef PROF
		gettimeofday(&t4, NULL);
		comp_time += 1000000 * (t4.tv_sec - t6.tv_sec) + (t4.tv_usec - t6.tv_usec);
#endif

#ifdef LOG_DATA
		LIFData *c_lif = (LIFData *)pNetCPU->ppNeurons[copy_idx];
		real *c_vm = c_lif->pV_m;
		int copy_size = c_gFiredTableSizes[currentIdx];

		log_array(v_file, c_vm, pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]);

		log_array(sim_file, c_gFiredTable + allNeuronNum * currentIdx, copy_size);

		// for (int i=0; i<pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]; i++) {
		// 	fprintf(v_file, "%.10lf \t", c_vm[i]);
		// }
		// fprintf(v_file, "\n");

		// for (int i=0; i<copySize; i++) {
		// 	fprintf(sim_file, "%d ", c_gFiredTable[allNeuronNum*currentIdx+i]);
		// }
		// fprintf(sim_file, "\n");

#endif

#if 1
		if (curr_delay >= minDelay - 1) {
#ifdef ASYNC
			MPI_Status status_t;
			int ret = MPI_Wait(&request_t, &status_t);
			assert(ret == MPI_SUCCESS);
#endif
			for (int d_ =0; d_ < minDelay; d_++) {
				int delay_idx = (time-minDelay+2+d_+maxDelay)%(maxDelay+1);
				for (int n_ = 0; n_ < node_num; n_++) {
					int start = cnd->_recv_start[n_*(minDelay+1)+d_];
					int end = cnd->_recv_start[n_*(minDelay+1)+d_+1];
					for (int i=start; i<end; i++) {
						c_gFiredTable[allNeuronNum*delay_idx + c_gFiredTableSizes[delay_idx] + i-start] = cnd->_recv_data[cnd->_recv_offset[n_]+i];
					}
					c_gFiredTableSizes[delay_idx] += end - start;
#ifdef LOG_DATA
					log_array_noendl(msg_file, cnd->_recv_data + cnd->_recv_offset[n_] + start, end - start); 
#endif
				}
#ifdef LOG_DATA
				fprintf(msg_file, "\n");
#endif
			}

#ifdef LOG_DATA
			log_cnd(cnd, time, send_file, recv_file);
#endif
			resetCND(cnd);
		}
#endif
#ifdef PROF
		gettimeofday(&t5, NULL);
		comm_time += 1000000 * (t5.tv_sec - t4.tv_sec) + (t5.tv_usec - t4.tv_usec);
#endif

	}
#ifdef DEBUG
		printf("\n");
#endif 
	gettimeofday(&te, NULL);

	double seconds =  te.tv_sec - ts.tv_sec + (te.tv_usec - ts.tv_usec)/1000000.0;

	printf("Thread %d Simulation finesed in %lfs\n", network->_nodeIdx, seconds);
#ifdef PROF
	printf("Thread %d Simulation perf %lf:%lf:%lf\n", network->_nodeIdx, comp_time, comm_time, sync_time);
#endif

	fclose(sim_file);
	fclose(v_file);
#ifdef LOG_DATA
	fclose(msg_file);
	fclose(send_file);
	fclose(recv_file);
	fclose(input_e_file);
	fclose(input_i_file);
#endif

	return 0;
}

int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd) {
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
	checkCudaErrors(cudaSetDevice(0));
	// print_mem("Before Network");

	GNetwork *pNetCPU = network->_network;
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);
	// print_mem("Copied Network");

	CrossNodeData * cnd_gpu = copyCNDtoGPU(cnd);
	// print_mem("Copied CND");

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->pConnection->nNum;
	int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("Thread %d, NeuronTypeNum: %d, SynapseTypeNum: %d\n", network->_nodeIdx, nTypeNum, sTypeNum);
	printf("Thread %d, NodeNeuronNum: %d, AllNeuronNum: %d, SynapseNum: %d\n", network->_nodeIdx, nodeNeuronNum, allNeuronNum, nodeSynapseNum);

	int maxDelay = pNetCPU->pConnection->maxDelay;
	int minDelay = pNetCPU->pConnection->minDelay;

	printf("Thread %d MaxDelay: %d MinDelay: %d\n", network->_nodeIdx, maxDelay,  minDelay);

	// print_mem("Before Buffers");

	GBuffers *buffers = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->pConnection->maxDelay, network->_dt);

	int *c_fired_sizes = NULL;
	checkCudaErrors(cudaMallocHost((void**)&c_fired_sizes, sizeof(int)*(maxDelay+1)));
	memset(c_fired_sizes, 0, sizeof(int)*(maxDelay+1));

	BlockSize *updateSize = getBlockSize(allNeuronNum, nodeSynapseNum);

	// print_mem("Alloced Buffers");

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
		printf( "Thread %d ntype %d: <<<%d, %d>>>\n", network->_nodeIdx, c_pNetGPU->pNTypes[i], updateSize[c_pNetGPU->pNTypes[i]].gridSize , updateSize[c_pNetGPU->pNTypes[i]].blockSize);
	}
	for (int i=0; i<sTypeNum; i++) {
		printf("Thread %d stype %d: <<<%d. %d>>>\n", network->_nodeIdx, c_pNetGPU->pSTypes[i], updateSize[c_pNetGPU->pSTypes[i]].gridSize, updateSize[c_pNetGPU->pSTypes[i]].blockSize);
	}

	int node_num = network->_nodeNum;

	int * c_g_idx2index = copyToGPU<int>(network->_crossnodeMap->_idx2index, allNeuronNum);
	int * c_g_cross_index2idx = copyToGPU<int>(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossSize);
	int * c_g_global_cross_data = gpuMalloc<int>(allNeuronNum * node_num);
	int * c_g_fired_n_num = gpuMalloc<int>(node_num);

	vector<int> firedInfo;
	double ts, te;
	ts = MPI_Wtime();

#ifdef PROF
	double t1, t2, t3, t4, t5, t6;
	double comp_time = 0, comm_time = 0, sync_time = 0;
	int *send_count = (int *)malloc(node_num * sizeof(int));
	int *recv_count = (int *)malloc(node_num * sizeof(int));
	memset(send_count, 0, node_num * sizeof(int));
	memset(recv_count, 0, node_num * sizeof(int));
#endif

	size_t fmem = 0, tmem = 0;
	checkCudaErrors(cudaMemGetInfo(&fmem, &tmem));
	printf("Thread %d, GPUMEM used: %lfGB\n", network->_nodeIdx, static_cast<double>((tmem - fmem)/1024.0/1024.0/1024.0));

	to_attach();

	for (int time=0; time<network->_simCycle; time++) {
#ifdef PROF
		t1 = MPI_Wtime();
#endif
		update_time<<<1, 1>>>(c_pNetGPU->pConnection, time, buffers->c_gFiredTableSizes);

		for (int i=0; i<nTypeNum; i++) {
			assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->pConnection, c_pNetGPU->ppNeurons[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSize[c_pNetGPU->pNTypes[i]]);
		}

#ifdef PROF
		t2 = MPI_Wtime();
		comp_time += t2-t1;
#endif
		int curr_delay = time % cnd->_min_delay;
		cudaGenerateCND(c_g_idx2index, c_g_cross_index2idx, cnd_gpu, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, max_delay, cnd_gpu->min_delay, node_num, time, (allNeuronNum+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

		// checkCudaErrors(cudaMemcpy(gCrossDataGPU->_firedNum + network->_nodeIdx * node_num, c_g_fired_n_num, sizeof(int)*node_num, cudaMemcpyDeviceToHost));
		MPI_Request request_t;
		update_cnd_gpu(cnd_gpu, cnd, curr_delay, &request_t);

#ifdef PROF
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

#ifdef PROF
		cudaDeviceSynchronize();
		t4 = MPI_Wtime();
		comp_time += t4 - t6;
#endif
		if (curr_delay >= minDelay -1) {
			MPI_Status status_t;
			int ret = MPI_Wait(&request_t, &status_t);
			assert(ret == MPI_SUCCESS);

			//int delay_idx = time % (maxDelay + 1);

			checkCudaErrors(cudaMemcpy(c_fired_sizes, buffers->c_gFiredTableSizes, sizeof(int)*(maxDelay+1), cudaMemcpyDeviceToHost));

			for (int d_=0; d_ < minDelay; d_++) {
				int delay_idx = (time-minDelay+2+d_+maxDelay)%(maxDelay+1);
				for (int n_ = 0; n_<node_num; n_++) {
					int start = cnd->_recv_start[n_*(minDelay+1)+d_];
					int end = cnd->_recv_start[n_*(minDelay+1)+d_+1];
					if (end > start) {
						assert(c_fired_sizes[delay_idx] + end - start <= allNeuronNum);
						checkCudaErrors(cudaMemcpy(buffers->c_gFiredTable + allNeuronNum*delay_idx + c_fired_sizes[delay_idx], cnd->_recv_data + cnd->_recv_offset[n_] + start, sizeof(int)*(end-start), cudaMemcpyHostToDevice));
						c_fired_sizes[delay_idx] += end - start;
					}
				}
			}
			checkCudaErrors(cudaMemcpy(buffers->c_gFiredTableSizes, c_fired_sizes, sizeof(int)*(maxDelay+1), cudaMemcpyHostToDevice));
#ifdef LOG_DATA
			log_cnd(cnd, time, send_file, recv_file);
#endif
			reset_cnd_gpu(cnd_gpu, cnd);
		}


#ifdef PROF
		cudaDeviceSynchronize();
		t5 = MPI_Wtime();
		comm_time += t5 - t4;
#endif

		// for (int i=0; i< node_num; i++) {
			// int i2idx = network->_nodeIdx + node_num * i;
			// if (gCrossDataGPU->_firedNum[i2idx] > 0) {
			// 	int num = gCrossDataGPU->_firedNum[i2idx];
			// 	cudaAddCrossNeurons<<<(num+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(c_pNetGPU->pConnection, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, gCrossDataGPU->_firedArrays[i2idx], gCrossDataGPU->_firedNum[i2idx], time);
			// }
		//}
		
		
#ifdef LOG_DATA
		for (int i=0; i<copySize; i++) {
			fprintf(sim_file, "%d ", buffers->c_neuronsFired[i]);
		}
		fprintf(sim_file, "\n");

		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf \t", c_vm[i]);
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
	for (int i=0; i<node_num; i++) {
		send += std::to_string(send_count[i]);
		send += ' ';
		recv += std::to_string(recv_count[i]);
		recv += ' ';
	}

	printf("Thread %d Data Send:%s\n", network->_nodeIdx, send.c_str());
	printf("Thread %d Data Recv:%s\n", network->_nodeIdx, recv.c_str());
#endif

	int *rate = (int*)malloc(sizeof(int)*nodeNeuronNum);
	copyFromGPU<int>(rate, buffers->c_gFireCount, nodeNeuronNum);

	char fire_filename[512];
	sprintf(fire_filename, "fire.gpu.mpi_%d.count", network->_nodeIdx); 
	FILE *rate_file = fopen(fire_filename, "w+");
	if (rate_file == NULL) {
		printf("Open file Sim.log failed\n");
		return 0;
	}

	for (int i=0; i<nodeNeuronNum; i++) {
		fprintf(rate_file, "%d \t", rate[i]);
	}

	// print_mem("Before Free");

	free(rate);
	fclose(rate_file);

	fclose(sim_file);
	fclose(v_file);

	free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);

	// print_mem("After Free");

	return 0;
}

