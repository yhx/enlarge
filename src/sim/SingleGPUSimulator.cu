/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <sys/time.h>
#include <stdio.h>
#include <iostream>

#include "../utils/utils.h"
#include "../utils/helper_c.h"
#include "../base/TypeFunc.h"
#include "../gpu_utils/helper_gpu.h"
// #include "../gpu_utils/gpu_utils.h"
#include "../gpu_utils/GBuffers.h"
#include "../gpu_utils/runtime.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
// #include "../gpu_utils/gpu_func.h"

#include "SingleGPUSimulator.h"

using std::cout;
using std::endl;


SingleGPUSimulator::SingleGPUSimulator(Network *network, real dt) : Simulator(network, dt)
{
}

SingleGPUSimulator::~SingleGPUSimulator()
{
}

int SingleGPUSimulator::run(real time, FireInfo &log)
{

	int sim_cycle = round(time/_dt);

	reset();

	SimInfo info(_dt);
	GNetwork *pNetCPU = _network->buildNetwork(info);

	FILE *v_file = fopen_c("v.gpu.log", "w+");
	FILE *input_e_file = fopen_c("input_e.gpu.log", "w+");
	FILE *input_i_file = fopen_c("input_i.gpu.log", "w+");
	FILE *ie_file = fopen_c("ie.gpu.log", "w+");
	FILE *ii_file = fopen_c("ii.gpu.log", "w+");
	FILE *fire_file = fopen_c("fire.gpu.log", "w+");
	FILE *log_file = fopen_c("sim.gpu.log", "w+");

	//findCudaDevice(0, NULL);
	checkCudaErrors(cudaSetDevice(0));
	GNetwork *c_pNetGPU = copyGNetworkToGPU(pNetCPU);

	int nTypeNum = c_pNetGPU->nTypeNum;
	int sTypeNum = c_pNetGPU->sTypeNum;
	int totalNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
	int totalSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];
	printf("NeuronTypeNum: %d, SynapseTypeNum: %d\n", nTypeNum, sTypeNum);
	printf("NeuronNum: %d, SynapseNum: %d\n", totalNeuronNum, totalSynapseNum);

	int maxDelay = pNetCPU->ppConnections[0]->maxDelay;
	printf("maxDelay: %d minDelay: %d\n", pNetCPU->ppConnections[0]->maxDelay, pNetCPU->ppConnections[0]->minDelay);


	GBuffers *buffers = alloc_buffers(totalNeuronNum, totalSynapseNum, pNetCPU->ppConnections[0]->maxDelay, _dt);

	BlockSize *updateSize = getBlockSize(totalNeuronNum, totalSynapseNum);

#ifdef LOG_DATA
	real *c_vm = hostMalloc<real>(totalNeuronNum);

	int copy_idx = getIndex(c_pNetGPU->pNTypes, nTypeNum, LIF);

	LIFData *c_g_lif = copyFromGPU<LIFData>(static_cast<LIFData *>(c_pNetGPU->ppNeurons[copy_idx]), 1);

	real *c_g_vm = c_g_lif->pV_m;
#ifdef DEBUG 
	real *c_g_ie = c_g_lif->pI_e;
	real *c_g_ii = c_g_lif->pI_i;
#endif
#endif

	// real *c_I_syn = hostMalloc<real>(totalSynapseNum);
	// int exp_idx = getIndex(c_pNetGPU->pSTypes, sTypeNum, Exp);
	// GExppSynapses *c_g_exp = copyFromGPU<GExppSynapses>(static_cast<GExppSynapses*>(c_pNetGPU->ppSynapses[exp_idx]), 1);
	// real *c_g_I_syn = c_g_exp->p_I_syn;

	// for (int i=0; i<nTypeNum; i++) {
	// 	cout << c_pNetGPU->pNTypes[i] << ": <<<" << updateSize[c_pNetGPU->pNTypes[i]].gridSize << ", " << updateSize[c_pNetGPU->pNTypes[i]].blockSize << ">>>" << endl;
	// }
	// for (int i=0; i<sTypeNum; i++) {
	// 	cout << c_pNetGPU->pSTypes[i] << ": <<<" << updateSize[c_pNetGPU->pSTypes[i]].gridSize << ", " << updateSize[c_pNetGPU->pSTypes[i]].blockSize << ">>>" << endl;
	// }

	size_t fmem = 0, tmem = 0;
	checkCudaErrors(cudaMemGetInfo(&fmem, &tmem));
	printf("GPUMEM used: %lfGB\n", static_cast<double>((tmem - fmem)/1024.0/1024.0/1024.0));

	vector<int> firedInfo;
	printf("Start runing for %d cycles\n", sim_cycle);
	struct timeval ts, te;
	gettimeofday(&ts, NULL);
	for (int time=0; time<sim_cycle; time++) {
		//printf("Cycle: %d ", time);
		//fflush(stdout);

#ifdef DEBUG
		copyFromGPU<real>(c_vm, buffers->c_gNeuronInput + c_pNetGPU->pNeuronNums[copy_idx], c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(input_e_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(input_e_file, "\n");
		copyFromGPU<real>(c_vm, buffers->c_gNeuronInput_I + c_pNetGPU->pNeuronNums[copy_idx], c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(input_i_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(input_i_file, "\n");
#endif

		update_time<<<1, 1>>>(buffers->c_gFiredTableSizes, maxDelay, time);

		for (int i=0; i<nTypeNum; i++) {
			cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[0], c_pNetGPU->ppNeurons[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, totalNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i],time, &updateSize[c_pNetGPU->pNTypes[i]]);
		}

		for (int i=0; i<sTypeNum; i++) {
			cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], buffers->c_gNeuronInput, buffers->c_gNeuronInput_I, buffers->c_gFiredTable, buffers->c_gFiredTableSizes, totalNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSize[c_pNetGPU->pSTypes[i]]);
		}


#ifdef LOG_DATA
		//LOG DATA
		int currentIdx = time%(maxDelay+1);

		uinteger_t copySize = 0;
		copyFromGPU(&copySize, buffers->c_gFiredTableSizes + currentIdx, 1);
		copyFromGPU(buffers->c_neuronsFired, buffers->c_gFiredTable + (totalNeuronNum*currentIdx), copySize);

		copyFromGPU<real>(c_vm, c_g_vm, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(v_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(v_file, "\n");
#ifdef DEBUG
		copyFromGPU<real>(c_vm, c_g_ie, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(ie_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(ie_file, "\n");
		copyFromGPU<real>(c_vm, c_g_ii, c_pNetGPU->pNeuronNums[copy_idx+1]-c_pNetGPU->pNeuronNums[copy_idx]);
		for (int i=0; i<c_pNetGPU->pNeuronNums[copy_idx+1] - c_pNetGPU->pNeuronNums[copy_idx]; i++) {
			fprintf(ii_file, "%.10lf \t", c_vm[i]);
		}
		fprintf(ii_file, "\n");
		// for (int i=0; i<c_pNetGPU->pSynapseNums[1] - c_pNetGPU->pSynapseNums[0]; i++) {
		// 	fprintf(dataFile, ", %lf", c_I_syn[i]);
		// }
#endif

		for (int i=0; i<copySize; i++) {
			fprintf(log_file, "%d ", buffers->c_neuronsFired[i]);
		}
		fprintf(log_file, "\n");

		//LOG SYNAPSE
		//copyFromGPU<int>(buffers->c_synapsesFired, buffers->c_gSynapsesLogTable, totalSynapseNum);
		//int synapseCount = 0;
		//if (time > 0) {
		//	for (int i=0; i<totalSynapseNum; i++) {
		//		if (buffers->c_synapsesFired[i] == time) {
		//			if (synapseCount ==  0) {
		//				if (copySize > 0) {
		//					fprintf(logFile, ", ");
		//				}
		//				fprintf(logFile, "%s", network->idx2sid[i].getInfo().c_str());
		//				synapseCount++;
		//			} else {
		//				fprintf(logFile, ", %s", network->idx2sid[i].getInfo().c_str());
		//			}
		//		}
		//	}
		//	fprintf(logFile, "\n");
		//}
#endif
	}
	cudaDeviceSynchronize();

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

	printf("Simulation finesed in %ld:%ld:%ld.%06lds\n", hours, minutes, seconds, uSeconds);

	//CALC Firing Rate
	int *rate = (int*)malloc(sizeof(int)*totalNeuronNum);
	copyFromGPU<int>(rate, buffers->c_gFireCount, totalNeuronNum);

	for (int i=0; i<totalNeuronNum; i++) {
		fprintf(fire_file, "%d \t", rate[i]);
	}
	fflush(fire_file);

	if (log.find("count") != log.end()) {

		log["count"].size = totalNeuronNum;
		log["count"].data = rate;

		//FILE *rateFile = fopen("GFire.log", "w+");
		//if (rateFile == NULL) {
		//	printf("ERROR: Open file Sim.log failed\n");
		//	return -1;
		//}

		//for (int i=0; i<totalNeuronNum; i++) {
		//	fprintf(rateFile, "%d \t", rate[i]);
		//}

		//fflush(rateFile);
		//fclose(rateFile);
	}

	if (log.find("Y") != log.end()) {
		real *Y = (real*)malloc(sizeof(real)*totalNeuronNum);
		copyFromGPU(Y, buffers->c_gXInput, totalNeuronNum);
		log["Y"].size = totalNeuronNum;
		log["Y"].data = Y;
	}

	if (log.find("X") != log.end()) {
		uinteger_t *X = (uinteger_t*)malloc(sizeof(uinteger_t)*totalNeuronNum);
		copyFromGPU(X, buffers->c_gLayerInput, totalNeuronNum);
		log["X"].size = totalNeuronNum;
		log["X"].data = X;
	}



	fclose_c(v_file);
	fclose_c(input_e_file);
	fclose_c(input_i_file);
	fclose_c(ie_file);
	fclose_c(ii_file);
	fclose_c(fire_file);
	fclose_c(log_file);

	free_buffers(buffers);
	freeGNetworkGPU(c_pNetGPU);
	freeGNetwork(pNetCPU);

	return 0;
}


int SingleGPUSimulator::runMultiNets(real time, int parts, FireInfo &log) {
	int sim_cycle = round(time/_dt);
	reset();

	checkCudaErrors(cudaSetDevice(0));

	// Network multiNet(_network, parts);
	_network->set_node_num(parts);
	SimInfo info(_dt);
	DistriNetwork *subnets = _network->buildNetworks(info);
	assert(subnets != NULL);
	// CrossThreadDataGPU *crossData = _network->arrangeCrossThreadDataGPU(parts);
	// assert(crossData != NULL);

	GNetwork ** networks = (GNetwork **)malloc(sizeof(GNetwork *) * parts);
	GBuffers **buffers = (GBuffers **)malloc(sizeof(GBuffers *) * parts);
	BlockSize **updateSizes = (BlockSize **)malloc(sizeof(GBuffers *) * parts);

	for (int i=0; i<parts; i++) {
		subnets[i]._simCycle = sim_cycle;
		subnets[i]._nodeIdx = i;
		subnets[i]._nodeNum = parts;
		subnets[i]._dt = _dt;

		GNetwork *pNetCPU = subnets[i]._network;
		networks[i] = copyGNetworkToGPU(pNetCPU);
		GNetwork *c_pNetGPU = networks[i];

		int nTypeNum = c_pNetGPU->nTypeNum;
		int sTypeNum = c_pNetGPU->sTypeNum;
		int nodeNeuronNum = c_pNetGPU->pNeuronNums[nTypeNum];
		int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
		int nodeSynapseNum = c_pNetGPU->pSynapseNums[sTypeNum];

		buffers[i] = alloc_buffers(allNeuronNum, nodeSynapseNum, pNetCPU->ppConnections[0]->maxDelay, _dt);
		updateSizes[i] = getBlockSize(allNeuronNum, nodeSynapseNum);

		printf("Subnet %d NeuronTypeNum: %d, SynapseTypeNum: %d\n", subnets[i]._nodeIdx, nTypeNum, sTypeNum);
		printf("Subnet %d NeuronNum: %d, SynapseNum: %d\n", subnets[i]._nodeIdx, nodeNeuronNum, nodeSynapseNum);
	}

	for (int time=0; time<sim_cycle; time++) {

		for (int p=0; p<parts; p++) {
			int allNeuronNum = subnets[p]._network->ppConnections[0]->nNum;
			GNetwork *c_pNetGPU = networks[p];
			update_time<<<1, 1>>>(buffers[p]->c_gFiredTableSizes, c_pNetGPU->ppConnections[0]->maxDelay, time);

			for (int i=0; i<c_pNetGPU->nTypeNum; i++) {
				assert(c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i] > 0);
				cudaUpdateType[c_pNetGPU->pNTypes[i]](c_pNetGPU->ppConnections[0], c_pNetGPU->ppNeurons[i], buffers[p]->c_gNeuronInput, buffers[p]->c_gNeuronInput_I, buffers[p]->c_gFiredTable, buffers[p]->c_gFiredTableSizes, allNeuronNum, c_pNetGPU->pNeuronNums[i+1]-c_pNetGPU->pNeuronNums[i], c_pNetGPU->pNeuronNums[i], time, &updateSizes[p][c_pNetGPU->pNTypes[i]]);
			}
		}

		for (int p=0; p<parts; p++) {
			int allNeuronNum = subnets[p]._network->ppConnections[0]->nNum;
			GNetwork *c_pNetGPU = networks[p];
			for (int i=0; i<c_pNetGPU->sTypeNum; i++) {
				assert(c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i] > 0);
				cudaUpdateType[c_pNetGPU->pSTypes[i]](c_pNetGPU->ppConnections[i], c_pNetGPU->ppSynapses[i], buffers[p]->c_gNeuronInput, buffers[p]->c_gNeuronInput_I, buffers[p]->c_gFiredTable, buffers[p]->c_gFiredTableSizes, allNeuronNum, c_pNetGPU->pSynapseNums[i+1]-c_pNetGPU->pSynapseNums[i], c_pNetGPU->pSynapseNums[i], time, &updateSizes[p][c_pNetGPU->pSTypes[i]]);
			}
		}
	}

	for (int i=0; i<parts; i++) {
		freeGNetworkGPU(networks[i]);
		free_buffers(buffers[i]);
	}

	free(networks);
	free(buffers);
	free(updateSizes);

	return 0;
}

