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
	int build_net();
	int save_net(const char *name);
	int load_net(const char *name);
	int distribute(DistriNetwork **, CrossNodeData **, SimInfo &, int);
	virtual int run(real time, bool gpu);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, bool gpu);
protected:
	int _node_id;
	int _node_num;
	DistriNetwork *_node_nets;
	CrossNodeData *_node_datas;
};

int run_node_cpu(DistriNetwork *network, CrossNodeData *cnd);
int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd);

#define ASYNC

#endif /* MULTINODESIMULATOR_H */

