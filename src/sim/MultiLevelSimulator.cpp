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
#include "../base/TypeFunc.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/msg_utils/msg_utils.h"
#include "../msg_utils/convert.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "MultiLevelSimulator.h"

MultiLevelSimulator::MultiLevelSimulator(Network *network, real dt) : Simulator(network, dt)
{
	_all_nets = NULL;
	_all_datas = NULL;
	_network_data = NULL;
	_data = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &_proc_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len = 0;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, _proc_id, _proc_num);
	_thread_num = 0;
}

MultiLevelSimulator::MultiLevelSimulator(const string &path, real dt) : Simulator(NULL, dt)
{
	_all_nets = NULL;
	_all_datas = NULL;
	_network_data = NULL;
	_data = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &_proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &_proc_num);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, _proc_id, _proc_num);
	printf("Data %s/%d\n", path.c_str(), _proc_id);
	// to_attach();
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

int MultiLevelSimulator::run(real time, int thread_num)
{
	FireInfo log;
	run(time, log, thread_num);
	return 0;
}

int MultiLevelSimulator::run(real time, FireInfo &log)
{
    printf("This api should not be called!\n");
	return -1;
}

int MultiLevelSimulator::build_net(int num, SplitType split, const char *name, const AlgoPara *para)
{

	SimInfo info(_dt);
	info.save_mem = true;

	if (!_all_nets) {
		assert(num > 1);

		_network->set_node_num(num);
		_all_nets = _network->buildNetworks(info, split, name, para);
		for (int i=0; i<num; i++) {
			_all_nets[i]._simCycle = 0;
			_all_nets[i]._nodeIdx = i;
			_all_nets[i]._nodeNum = num;
			_all_nets[i]._dt = _dt;
		}
	}

	if (!_all_datas) {
		_all_datas = _network->arrangeCrossNodeData(info);
	}

	return 0;
}

int MultiLevelSimulator::save_net(const string &path)
{
	mkdir(path.c_str());
	string name = path + "/meta.data";
	FILE *f = fopen_c(name.c_str(), "w");

	int num = _proc_num * _thread_num;
	fwrite_c(&(num), 1, f);
	fclose_c(f);

	if (_network && _data) {
		for (int i=0; i<_thread_num; i++) {
			int idx = _proc_id * _thread_num + i;
			string path_i = path + "/" + std::to_string(idx);
			mkdir(path_i.c_str());
			saveDistriNet(_network_data[i], path_i);
			saveCND(_data[i], path_i);
		}
	} else if (_all_nets && _all_datas) {
		for (int i=0; i<_proc_num * _thread_num; i++) {
			string path_i = path + "/" + std::to_string(i);
			mkdir(path_i.c_str());
			saveDistriNet(&(_all_nets[i]), path_i);
			saveCND(&(_all_datas[i]), path_i);
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
	int num;
	FILE *f = fopen_c(name.c_str(), "r");
	fread_c(&(num), 1, f);
	fclose_c(f);

	if (num != _proc_num * _thread_num) {
		printf("Error: instance num mismatch network data\n");
		exit(-1);
	}

	_network_data = malloc_c<DistriNetwork *>(_thread_num);
	_data = malloc_c<CrossNodeData *>(_thread_num);

	for (int i=0; i<_thread_num; i++) {
		string path_i = path + "/" + std::to_string(_proc_id * _thread_num + i);
		_network_data[i] = loadDistriNet(path_i);
		_data[i] = loadCND(path_i);
	}
	
	return 0;
}

int MultiLevelSimulator::distribute(SimInfo &info, int sim_cycle)
{

	printf("Distritubing Network\n");
	int num = _proc_num * _thread_num;
	if (_proc_id == 0) {
		// print_mem("Finish Network");
		// print_mem("Finish CND");
		if (!(_all_nets && _all_datas)) {
			build_net(num);
		}

		for (int i=0; i<num; i++) {
			_all_nets[i]._simCycle = sim_cycle;
			_all_nets[i]._nodeIdx = i;
			_all_nets[i]._nodeNum = num;
			_all_nets[i]._dt = _dt;
		}
	} else {
		printf("Rank %d, Wait for network build\n", _proc_id);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	_network_data = malloc_c<DistriNetwork *>(_thread_num);
	_data = malloc_c<CrossNodeData *>(_thread_num);

	if (_proc_id == 0) {
		for (int i=0; i<_thread_num; i++) {
			_network_data[i] = &(_all_nets[i]);
			_data[i] = &(_all_datas[i]);
		}
		// allocDataCND(_data);
		// print_mem("AllocData CND");

		for (int p=1; p<_proc_num; p++) {
			int tag = DATA_TAG;
			for (int t=0; t<_thread_num; t++) {
				int dst = p;
				int i = p * _thread_num + t;
				printf("Send to %d, tag: %d\n", dst, tag);
				sendDistriNet(&(_all_nets[i]), dst, tag, MPI_COMM_WORLD);
				printf("Send DistriNet to %d, tag: %d\n", dst, tag);
				tag += DNET_TAG;
				sendCND(&(_all_datas[i]), dst, tag, MPI_COMM_WORLD);
				printf("Send CND to %d, tag: %d\n", dst, tag);
				tag += CND_TAG;
			}
		}
		// network = initDistriNet(1, _dt);
		// network->_network = _network->buildNetwork(info);
		// network->_simCycle = sim_cycle;
		// network->_nodeIdx = 0;
		// network->_nodeNum = proc_num;
		// network->_dt = _dt;
		// data = NULL;
	} else {
		int tag = DATA_TAG;
		for (int t=0; t<_thread_num; t++) {
			printf("%d recv from %d, tag: %d\n", _proc_id, 0, tag);
			_network_data[t] = recvDistriNet(0, tag, MPI_COMM_WORLD);
			printf("%d recv DistriNet from %d, tag: %d\n", _proc_id, 0, tag);
			tag += DNET_TAG;
			_data[t] = recvCND(0, tag, MPI_COMM_WORLD);
			printf("%d recv CND from %d, tag: %d\n", _proc_id, 0, tag);
			tag += CND_TAG;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
}

int MultiLevelSimulator::run(real time, FireInfo &log, int thread_num)
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

	CrossMap **cm = malloc_c<CrossMap*>(_thread_num);
	CrossSpike **cs = malloc_c<CrossSpike*>(_thread_num);

	for (int i=0; i<_thread_num; i++) {
		if (_network_data[i]->_simCycle != sim_cycle) {
			_network_data[i]->_simCycle = sim_cycle;
		}
		cm[i] = convert2crossmap(_network_data[i]->_crossnodeMap);
	    cs[i] = convert2crossspike(_data[i], _proc_id, _thread_num);
	}

	

	assert(thread_num > 0);
	pthread_barrier_init(&g_proc_barrier, NULL, thread_num);
	pthread_t *thread_ids = malloc_c<pthread_t>(thread_num);
	assert(thread_ids != NULL);

	RunPara *paras = malloc_c<RunPara>(thread_num);

	for (int i=0; i<thread_num; i++) {
		paras[i]._net = _network_data[i];
		paras[i]._cm = cm[i];
		paras[i]._cs = cs;
		paras[i]._thread_id = i;

		int ret = pthread_create(&(thread_ids[i]), NULL, &run_thread_ml, (void*)&(paras[i]));
		assert(ret == 0);
	}

	for (int i=0; i<thread_num; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	pthread_barrier_destroy(&g_proc_barrier);
	// if (gpu > 0) {
	// 	run_proc_gpu(_network_data, map, msg);
	// } else {
	// 	run_proc_cpu(_network_data, map, msg);
	// }
	
	free_c(paras);

	for (int i=0; i<_thread_num; i++) {
		delete cm[i];
		delete cs[i];
	}

	free_c(cm);
	free_c(cs);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

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
	// int nodeNeuronNum = pNetCPU->pNeuronNums[nTypeNum];
	int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
	// int nodeSynapseNum = pNetCPU->pSynapseNums[sTypeNum];

	int max_delay = pNetCPU->ppConnections[0]->maxDelay;
	// int min_delay = pNetCPU->ppConnections[0]->minDelay;
	// assert(min_delay == msg->_min_delay);

	pInfoGNetwork(pNetCPU, string("Proc ") + std::to_string(network->_nodeIdx)); 

	int proc_num = msg->_proc_num;

	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, max_delay);

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
		int currentIdx = time % (max_delay+1);
		buffer._fired_sizes[currentIdx] = 0;

#ifdef LOG_DATA
		log_array(input_e_file, buffer._data, pNetCPU->bufferOffsets[nTypeNum]);
#endif

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i], buffer._data, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

#if 1
		// memset(cnd->_send_num, 0, sizeof(int)*(proc_num));
#ifdef PROF
		gettimeofday(&t2, NULL);
		comp_time += 1000000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec);
#endif
		// int curr_delay = time % msg->_min_delay;

		msg->fetch_cpu(map, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, proc_num, max_delay, time);

		msg->update_cpu(time);

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
#endif

#ifdef LOG_DATA
		msg->log_cpu(time, (string("proc_") + std::to_string(network->_nodeIdx)).c_str());
#endif
		msg->upload_cpu(buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, max_delay, time);

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

