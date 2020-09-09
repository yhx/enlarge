/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef MULTINODESIMULATOR_H
#define MULTINODESIMULATOR_H

#include "../interface/Simulator.h"

class MultiNodeSimulator : public Simulator {
public:
	MultiNodeSimulator(Network *network, real dt);
	~MultiNodeSimulator();

	using Simulator::run;
	int mpi_init(int *argc, char ***argv);
	virtual int run(real time, bool gpu);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, bool gpu);
protected:
	int node_id;
	int node_num;
};

#endif /* MULTINODESIMULATOR_H */

