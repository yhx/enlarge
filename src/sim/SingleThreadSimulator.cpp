/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../utils/utils.h"
#include "../utils/helper_c.h"
#include "../base/TypeFunc.h"
#include "../neuron/lif/LIFData.h"

#include "SingleThreadSimulator.h"

int *c_gFiredCount = NULL;

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
	FILE *fire_file = fopen_c("fire.cpu.log", "w+");
#ifdef LOG_DATA
	FILE *input_e_file = fopen_c("input_e.cpu.log", "w+");
	FILE *input_i_file = fopen_c("input_i.cpu.log", "w+");
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

	int cFiredTableCap = totalNeuronNum;



	real *c_gNeuronInput = (real*)malloc(sizeof(real)*totalNeuronNum);
	memset(c_gNeuronInput, 0, sizeof(real)*totalNeuronNum);
	real *c_gNeuronInput_I = (real*)malloc(sizeof(real)*totalNeuronNum); 
	memset(c_gNeuronInput_I, 0, sizeof(real)*totalNeuronNum);
	uinteger_t *c_gFiredTable = malloc_c<uinteger_t>(totalNeuronNum*(maxDelay+1));
   	uinteger_t *c_gFiredTableSizes = malloc_c<uinteger_t>(maxDelay+1);
   	memset(c_gFiredTableSizes, 0, sizeof(int)*(maxDelay+1));

   	c_gFiredCount = (int*)malloc(sizeof(int)*(totalNeuronNum));
   	memset(c_gFiredCount, 0, sizeof(int)*(totalNeuronNum));

	printf("Start runing for %d cycles\n", sim_cycle);
	struct timeval ts, te;
	gettimeofday(&ts, NULL);

	for (int time=0; time<sim_cycle; time++) {
		//printf("\rCycle: %d", cycle);
		//fflush(stdout);
#ifdef LOG_DATA
		int copy_idx = getIndex(pNetCPU->pNTypes, nTypeNum, LIF);

		log_array(input_e_file, c_gNeuronInput, totalNeuronNum);
		log_array(input_i_file, c_gNeuronInput_I, totalNeuronNum);

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
		c_gFiredTableSizes[currentIdx] = 0;

		for (int i=0; i<nTypeNum; i++) {
			updateType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i], pNetCPU->pNeuronNums[i], time);
		}

		for (int i=0; i<sTypeNum; i++) {
			updateType[pNetCPU->pSTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppSynapses[i], c_gNeuronInput, c_gNeuronInput_I, c_gFiredTable, c_gFiredTableSizes, cFiredTableCap, pNetCPU->pSynapseNums[i+1]-pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time);
		}

#ifdef LOG_DATA
		LIFData *c_lif = (LIFData *)pNetCPU->ppNeurons[copy_idx];
		real *c_vm = c_lif->pV_m;

		int copy_size = c_gFiredTableSizes[currentIdx];

		log_array(v_file, c_vm, pNetCPU->pNeuronNums[copy_idx+1] - pNetCPU->pNeuronNums[copy_idx]);

		log_array(log_file, c_gFiredTable + totalNeuronNum * currentIdx, copy_size);

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

	for (int i=0; i<totalNeuronNum; i++) {
		fprintf(fire_file, "%d \t", c_gFiredCount[i]); 
	}



	fclose(v_file);
	// fclose(input_e_file);
	// fclose(input_i_file);
	// fclose(ie_file);
	// fclose(ii_file);
	fclose(fire_file);
	fclose(log_file);

	return 0;
}
