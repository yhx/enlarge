/* This program is writen by lh21.
 * usually just for fun.
 */

#ifndef MULTITHREADSIMULATOR_H
#define MULTITHREADSIMULATOR_H

#include "../interface/Simulator.h"

class MultiThreadSimulator : public Simulator {
public:
	MultiThreadSimulator(Network *network, real dt);
	~MultiThreadSimulator();

	using Simulator::run;

	virtual int run(real time, int thread_num);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, int thread_num);
};

/**
 * @brief data structure for pthread parameters 
 */
struct MTPthreadPara {
	int _sim_cycle;
	GNetwork *_pNetCPU;
	SimInfo *_info;

	int _nTypeNum;
	int _sTypeNum;
	int _totalNeuronNum;
	int _totalSynapseNum;
	int _maxDelay;
	Buffer *_buffer;

	size_t _thread_num;
	size_t _thread_id;
};

void *multi_thread_cpu(void *paras);

#endif /* SINGLETHREADSIMULATOR_H */

