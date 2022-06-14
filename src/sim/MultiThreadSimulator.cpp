/* This program is writen by lh21.
 * usually just for fun.
 */

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <atomic>

#include "../utils/utils.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../utils/Buffer.h"
#include "../base/TypeFunc.h"
#include "../neuron/lif/LIFData.h"

#include "MultiThreadSimulator.h"

MultiThreadSimulator::MultiThreadSimulator(Network *network, real dt)
	: Simulator(network, dt)
{
}

MultiThreadSimulator::~MultiThreadSimulator()
{
}

int MultiThreadSimulator::run(real time, int thread_num) {
	FireInfo log;
	run(time, log, thread_num);
	return 0;
}

int MultiThreadSimulator::run(real time, FireInfo &log) {
	int thread_num = 32;
	run(time, log, thread_num);
	return 0;
}


pthread_barrier_t multi_thread_proc_barrier;


int MultiThreadSimulator::run(real time, FireInfo &log, int thread_num)
{
	int sim_cycle = round((time)/_dt);
	reset();

	SimInfo info(_dt);
	GNetwork *pNetCPU = _network->buildNetwork(info);

	int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
	int totalNeuronNum = pNetCPU->pNeuronNums[nTypeNum];
	int totalSynapseNum = pNetCPU->pSynapseNums[sTypeNum];
	
	printf("NeuronTypeNum: %d, SynapseTypeNum: %d\n", nTypeNum, sTypeNum);
	printf("NeuronNum: %d, SynapseNum: %d\n", totalNeuronNum, totalSynapseNum);

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	printf("maxDelay: %d minDelay: %d\n", pNetCPU->ppConnections[0]->maxDelay, pNetCPU->ppConnections[0]->minDelay);
	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], totalNeuronNum, maxDelay);

	printf("thread_num: %d\n", thread_num);

	assert(thread_num > 0);
	pthread_barrier_init(&multi_thread_proc_barrier, NULL, thread_num);
	pthread_t *thread_ids = malloc_c<pthread_t>(thread_num);
	assert(thread_ids != NULL);
	
	MTPthreadPara *paras = malloc_c<MTPthreadPara>(thread_num);

	for (int i = 0; i < thread_num; ++i) {
		printf("start thread %d\n", i);
		paras[i]._sim_cycle = sim_cycle;
		paras[i]._pNetCPU = pNetCPU;
		paras[i]._info = &info;
		paras[i]._nTypeNum = nTypeNum;
		paras[i]._sTypeNum = sTypeNum;
		paras[i]._totalNeuronNum = totalNeuronNum;
		paras[i]._totalSynapseNum = totalSynapseNum;
		paras[i]._maxDelay = maxDelay;
		paras[i]._buffer = &buffer;
		paras[i]._thread_num = thread_num;
		paras[i]._thread_id = i;
		int ret = pthread_create(&(thread_ids[i]), NULL, multi_thread_cpu, (void*)&(paras[i]));
		assert(ret == 0);
	}

	for (int i = 0; i < thread_num; ++i) {
		pthread_join(thread_ids[i], NULL);
	}
	pthread_barrier_destroy(&multi_thread_proc_barrier);

	free_c(paras);

	return 0;
}

void *multi_thread_cpu(void *paras) {
	MTPthreadPara *tmp = static_cast<MTPthreadPara*>(paras);
	int sim_cycle = tmp->_sim_cycle;
	GNetwork *pNetCPU = tmp->_pNetCPU;
	SimInfo *info = tmp->_info;
	int nTypeNum = tmp->_nTypeNum;
	int sTypeNum = tmp->_sTypeNum;
	int totalNeuronNum = tmp->_totalNeuronNum;
	int totalSynapseNum = tmp->_totalSynapseNum;
	int maxDelay = tmp->_maxDelay;
	Buffer &buffer = *tmp->_buffer;
	size_t thread_num = tmp->_thread_num;
	size_t thread_id = tmp->_thread_id;
	
	FILE *v_file, *log_file;
#ifdef LOG_DATA
	FILE *input_file;
#endif

	if (thread_id == 0) {
		v_file = fopen_c("v.cpu.log", "w+");
		log_file = fopen_c("sim.cpu.log", "w+");
#ifdef LOG_DATA
		input_file = fopen_c("input.cpu.log", "w+");
#endif
	}

	if (thread_id == 0) {
		printf("Start runing for %d cycles\n", sim_cycle);
	}
	struct timeval ts, te;
	gettimeofday(&ts, NULL);

	for (int time = 0; time < sim_cycle; time++) {
		if (thread_id == 0) {
			printf("Time: %d\n", time);
			// fflush(stdout);
		}

#ifdef LOG_DATA
		log_array(input_file, buffer._data, pNetCPU->bufferOffsets[nTypeNum]);
#endif

		info->currCycle = time;
		info->fired.clear();
		info->input.clear();

		int currentIdx = time % (maxDelay + 1);
		buffer._fired_sizes[currentIdx] = 0;

		for (int i = 0; i < nTypeNum; i++) {
			updatePthreadType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i],
			 	buffer._data + pNetCPU->bufferOffsets[i], buffer._fire_table, buffer._fired_sizes, 
				buffer._fire_table_cap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i],
				pNetCPU->pNeuronNums[i], time, thread_num, thread_id, multi_thread_proc_barrier);
		}

		for (int i = 0; i < sTypeNum; i++) {
			updatePthreadType[pNetCPU->pSTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppSynapses[i], 
				buffer._data, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap,
				pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time, thread_num,
				thread_id, multi_thread_proc_barrier);
		}

		pthread_barrier_wait(&multi_thread_proc_barrier);
#ifdef LOG_DATA
		for (size_t i = 0; i < nTypeNum; i++) {
			real *c_vm = getVNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i]);
			log_array(v_file, c_vm, pNetCPU->pNeuronNums[i+1] - pNetCPU->pNeuronNums[i]);
		}
		
		int copy_size = buffer._fired_sizes[currentIdx];

		log_array(log_file, buffer._fire_table  + totalNeuronNum * currentIdx, copy_size);
#endif
	}

	pthread_barrier_wait(&multi_thread_proc_barrier);

	gettimeofday(&te, NULL);
	long seconds = te.tv_sec - ts.tv_sec;
	long hours = seconds / 3600;
	seconds = seconds % 3600;
	long minutes = seconds / 60;
	seconds = seconds % 60;
	long uSeconds = te.tv_usec - ts.tv_usec;
	if (uSeconds < 0) {
		uSeconds += 1000000;
		seconds = seconds - 1;
	}

	printf("Thread %ld's simulation finished in %ld:%ld:%ld.%06lds\n", thread_id, hours, minutes, seconds, uSeconds);

	if (thread_id == 0) {
		for (int i = 0; i < nTypeNum; i++) {
			logRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], "cpu");
		}

		fclose(v_file);
		fclose(log_file);

#ifdef LOG_DATA
		fclose(input_file);
#endif
	}

	return 0;
}
