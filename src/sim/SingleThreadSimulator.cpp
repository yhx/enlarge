/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../utils/utils.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../utils/Buffer.h"
#include "../base/TypeFunc.h"
#include "../neuron/lif/LIFData.h"

#include "SingleThreadSimulator.h"

SingleThreadSimulator::SingleThreadSimulator(Network *network, real dt)
	: Simulator(network, dt)
{
}

SingleThreadSimulator::~SingleThreadSimulator()
{
}

int SingleThreadSimulator::run(real time, FireInfo &log)
{
	int sim_cycle =  round((time)/_dt);

	reset();

	SimInfo info(_dt);

	GNetwork *pNetCPU = _network->buildNetwork(info);

	FILE *v_file = fopen_c("v.cpu.log", "w+");
	FILE *log_file = fopen_c("sim.cpu.log", "w+");
	// FILE *fire_file = fopen_c("fire.cpu.log", "w+");
#ifdef LOG_DATA
	FILE *input_file = fopen_c("input.cpu.log", "w+");
	// FILE *ie_file = fopen_c("ie.cpu.log", "w+");
	// FILE *ii_file = fopen_c("ii.cpu.log", "w+");
#endif

	int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
	int totalNeuronNum = pNetCPU->pNeuronNums[nTypeNum];
	int totalSynapseNum = pNetCPU->pSynapseNums[sTypeNum];
	printf("NeuronTypeNum: %d, SynapseTypeNum: %d\n", nTypeNum, sTypeNum);
	printf("NeuronNum: %d, SynapseNum: %d\n", totalNeuronNum, totalSynapseNum);

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	// int deltaDelay = pNetCPU->ppConnections->maxDelay - pNetCPU->ppConnections->minDelay + 1;
	printf("maxDelay: %d minDelay: %d\n", pNetCPU->ppConnections[0]->maxDelay, pNetCPU->ppConnections[0]->minDelay);

	Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], totalNeuronNum, maxDelay);

	printf("Start runing for %d cycles\n", sim_cycle);
	struct timeval ts, te;
	gettimeofday(&ts, NULL);

	for (int time=0; time<sim_cycle; time++) {
		//printf("\rCycle: %d", cycle);
		//fflush(stdout);
#ifdef LOG_DATA
		log_array(input_file, buffer._data, pNetCPU->bufferOffsets[nTypeNum]);

		// for (int i=pNetCPU->pNeuronNums[copy_idx]; i<pNetCPU->pNeuronNums[copy_idx+1]; i++) {
		// 	fprintf(input_e_file, "%.10lf \t", c_gNeuronInput[i]);
		// }
		// fprintf(input_e_file, "\n");
		// for (int i=pNetCPU->pNeuronNums[copy_idx]; i<pNetCPU->pNeuronNums[copy_idx+1]; i++) {
		// 	fprintf(input_i_file, "%.10lf \t", c_gNeuronInput_I[i]);
		// }
		// fprintf(input_i_file, "\n");
#endif

		info.currCycle = time;
		info.fired.clear();
		info.input.clear();

		int currentIdx = time % (maxDelay + 1);
		buffer._fired_sizes[currentIdx] = 0;

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i], buffer._data + pNetCPU->bufferOffsets[i], buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

		for (int i=0; i<sTypeNum; i++) {
			updateType[pNetCPU->pSTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppSynapses[i], buffer._data, buffer._fire_table, buffer._fired_sizes, buffer._fire_table_cap, pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time);
		}

#ifdef LOG_DATA
		// int copy_idx = getIndex(pNetCPU->pNTypes, nTypeNum, LIF);
		// LIFData *c_lif = (LIFData *)pNetCPU->ppNeurons[copy_idx];
		for (size_t i = 0; i < nTypeNum; i++) {
			real *c_vm = getVNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i]);
			log_array(v_file, c_vm, pNetCPU->pNeuronNums[i+1] - pNetCPU->pNeuronNums[i]);
		}
		
		// real *c_vm = c_lif->pV_m;

		int copy_size = buffer._fired_sizes[currentIdx];

		// log_array(v_file, c_vm, pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]);

		log_array(log_file, buffer._fire_table  + totalNeuronNum * currentIdx, copy_size);

		// for (int i=0; i<pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]; i++) {
		// 	fprintf(v_file, "%.10lf \t", c_vm[i]);
		// }
		// fprintf(v_file, "\n");

		// for (int i=0; i<copy_size; i++) {
		// 	fprintf(log_file, "%d ", c_gFiredTable[totalNeuronNum*currentIdx+i]);
		// }
		// fprintf(log_file, "\n");
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

	printf("\nSimulation finished in %ld:%ld:%ld.%06lds\n", hours, minutes, seconds, uSeconds);

	for (int i=0; i<nTypeNum; i++) {
		logRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], "cpu");
	}



	fclose(v_file);
	// fclose(input_i_file);
	// fclose(ie_file);
	// fclose(ii_file);
	fclose(log_file);

#ifdef LOG_DATA
	fclose(input_file);
#endif

	return 0;
}
