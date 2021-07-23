/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <mpi.h>

#include "../utils/utils.h"
#include "../utils/helper_c.h"
#include "../base/TypeFunc.h"
#include "../msg_utils/msg_utils.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "MultiLevelSimulator.h"

MultiLevelSimulator::MultiLevelSimulator(Network *network, real dt) : Simulator(network, dt)
{
	_proc_nets = NULL;
	_proc_datas = NULL;
	_network_data = NULL;
	_data = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &_proc_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, _proc_id, _proc_num);
}

MultiLevelSimulator::MultiLevelSimulator(const string &path, real dt) : Simulator(NULL, dt)
{
	_proc_nets = NULL;
	_proc_datas = NULL;
	_network_data = NULL;
	_data = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &_proc_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, _proc_id, _proc_num);
	printf("Data %s/%d\n", path.c_str(), _proc_id);
	to_attach();
	load_net(path);
}

MultiLevelSimulator::~MultiLevelSimulator()
{
}

int MultiLevelSimulator::mpi_init(int *argc, char ***argv)
{
	MPI_Init(argc, argv);
	return 0;
}

int MultiLevelSimulator::run(real time, FireInfo &log)
{
	run(time, log, 1);
	return 0;
}

int MultiLevelSimulator::build_net(int num, SplitType split, const char *name, const AlgoPara *para)
{

	SimInfo info(_dt);
	info.save_mem = true;

	if (!_proc_nets) {
		if (num <= 1) {
			num = _proc_num;
		} else {
			_proc_num = num;
		}
		_network->set_node_num(num);
		_proc_nets = _network->buildNetworks(info, split, name, para);
		for (int i=0; i<_proc_num; i++) {
			_proc_nets[i]._simCycle = 0;
			_proc_nets[i]._nodeIdx = i;
			_proc_nets[i]._nodeNum = _proc_num;
			_proc_nets[i]._dt = _dt;
		}
	}

	if (!_proc_datas) {
		_proc_datas = _network->arrangeCrossNodeData(info);
	}

	return 0;
}

int MultiLevelSimulator::save_net(const string &path)
{
	mkdir(path.c_str());
	string name = path + "/meta.data";
	FILE *f = fopen_c(name.c_str(), "w");
	fwrite_c(&(_proc_num), 1, f);
	fclose_c(f);

	if (_network && _data) {
			string path_i = path + "/" + std::to_string(_proc_id);
			mkdir(path_i.c_str());
			saveDistriNet(_network_data, path_i);
			saveCND(_data, path_i);
	} else if (_proc_nets && _proc_datas) {
		for (int i=0; i<_proc_num; i++) {
			string path_i = path + "/" + std::to_string(i);
			mkdir(path_i.c_str());
			saveDistriNet(&(_proc_nets[i]), path_i);
			saveCND(&(_proc_datas[i]), path_i);
		}
	} else {
		printf("Before save, build the net first\n");
		return 1;
	}

	return 0;
}

int MultiLevelSimulator::load_net(const string &path)
{
	string name = path + "/meta.data";
	int node_num;
	FILE *f = fopen_c(name.c_str(), "r");
	fread_c(&(node_num), 1, f);
	fclose_c(f);

	if (_proc_num != node_num) {
		printf("Error: node num mismatch network data\n");
		exit(-1);
	}

	string path_i = path + "/" + std::to_string(_proc_id);
	_network_data = loadDistriNet(path_i);
    _data = loadCND(path_i);
	
	return 0;
}

int MultiLevelSimulator::distribute(SimInfo &info, int sim_cycle)
{

	printf("Distritubing Network\n");
	if (_proc_id == 0) {
		// print_mem("Finish Network");
		// print_mem("Finish CND");
		if (!(_proc_nets && _proc_datas)) {
			build_net();
		}

		for (int i=0; i<_proc_num; i++) {
			_proc_nets[i]._simCycle = sim_cycle;
			_proc_nets[i]._nodeIdx = i;
			_proc_nets[i]._nodeNum = _proc_num;
			_proc_nets[i]._dt = _dt;
		}
	} else {
		printf("Rank %d, Wait for network build\n", _proc_id);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (_proc_id == 0) {
		_network_data = &(_proc_nets[0]);
		_data = &(_proc_datas[0]);
		// allocDataCND(_data);
		// print_mem("AllocData CND");

		for (int i=1; i<_proc_num; i++) {
			printf("Send to %d, tag: %d\n", i, DATA_TAG);
			sendDistriNet(&(_proc_nets[i]), i, DATA_TAG, MPI_COMM_WORLD);
			printf("Send DistriNet to %d, tag: %d\n", i, DATA_TAG);
			sendCND(&(_proc_datas[i]), i, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
			printf("Send CND to %d, tag: %d\n", i, DATA_TAG);
		}
		// network = initDistriNet(1, _dt);
		// network->_network = _network->buildNetwork(info);
		// network->_simCycle = sim_cycle;
		// network->_nodeIdx = 0;
		// network->_nodeNum = node_num;
		// network->_dt = _dt;
		// data = NULL;
	} else {
		printf("%d recv from %d, tag: %d\n", _proc_id, 0, DATA_TAG);
		_network_data = recvDistriNet(0, DATA_TAG, MPI_COMM_WORLD);
		printf("%d recv DistriNet from %d, tag: %d\n", _proc_id, 0, DATA_TAG);
		_data = recvCND(0, DATA_TAG + DNET_TAG, MPI_COMM_WORLD);
		printf("%d recv CND from %d, tag: %d\n", _proc_id, 0, DATA_TAG);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
}

int MultiLevelSimulator::run(real time, FireInfo &log, bool gpu)
{

	int sim_cycle = round(time/_dt);
	reset();

	SimInfo info(_dt);
	info.save_mem = true;


	to_attach();

	printf("Rank %d, before distribute\n", _proc_id);

	if (!(_network_data && _data)) {
		distribute(info, sim_cycle);
	} 

	if (_network_data->_simCycle != sim_cycle) {
		_network_data->_simCycle = sim_cycle;
	}

	if (gpu) {
		run_proc_gpu(_network_data, _data);
	} else {
		run_proc_cpu(_network_data, _data);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	M296 465 573PI_Finalize();

	return 0;
}

int run_proc_cpu(DistriNetwork *network, CrossMap *map, CrossSpike *msg) {
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

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	int minDelay = pNetCPU->ppConnections[0]->minDelay;
	assert(minDelay == cnd->_min_delay);

	pInfoGNetwork(pNetCPU, string("Proc ") + network->_nodeIdx); 

	int node_num = cnd->_proc_num;

	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, maxDelay);

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
		int currentIdx = time % (maxDelay+1);
		buffer._fired_sizes[currentIdx] = 0;

#ifdef LOG_DATA
		log_array(input_e_file, buffer._data, pNetCPU->bufferOffsets[nTypeNum]);
#endif

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i], buffer._data, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

#if 1
		// memset(cnd->_send_num, 0, sizeof(int)*(node_num));
#ifdef PROF
		gettimeofday(&t2, NULL);
		comp_time += 1000000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec);
#endif
		int curr_delay = time % cnd->_min_delay;
		cs.fetch_cpu(map, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, _proc_num, 
		generateCND(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, cnd, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, maxDelay, cnd->_min_delay, network->_nodeNum, time);


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
			updateType[pNetCPU->pSTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppSynapses[i], buffer._data, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time);
		}

#ifdef PROF
		gettimeofday(&t4, NULL);
		comp_time += 1000000 * (t4.tv_sec - t6.tv_sec) + (t4.tv_usec - t6.tv_usec);
#endif

#ifdef LOG_DATA
		LIFData *c_lif = (LIFData *)pNetCPU->ppNeurons[copy_idx];
		real *c_vm = c_lif->pV_m;
		int copy_size = buffer._fired_sizes[currentIdx];

		log_array(v_file, c_vm, pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]);

		log_array(sim_file, buffer._fire_table + allNeuronNum * currentIdx, copy_size);

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
						buffer._fire_table[allNeuronNum*delay_idx + buffer._fired_sizes[delay_idx] + i-start] = cnd->_recv_data[cnd->_recv_offset[n_]+i];
					}
					buffer._fired_sizes[delay_idx] += end - start;
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

	char name[512];
	sprintf(name, "mpi_%d", network->_nodeIdx); 

	for (int i=0; i<nTypeNum; i++) {
		logRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], name);
	}


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

