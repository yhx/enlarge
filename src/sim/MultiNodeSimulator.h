/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef MULTINODESIMULATOR_H
#define MULTINODESIMULATOR_H

#include <string>

#include "../interface/Simulator.h"

using std::string;

class MultiNodeSimulator : public Simulator {
public:
	MultiNodeSimulator(const string &path, real dt);

	MultiNodeSimulator(Network *network, real dt);
	~MultiNodeSimulator();

	using Simulator::run;
	int mpi_init(int *argc, char ***argv);
	int build_net(int num = 0, SplitType split=SynapseBalance, const char *name="", const AlgoPara *para = NULL);
	int save_net(const string &name);
	int load_net(const string &name);
	int distribute(SimInfo &, int);
	virtual int run(real time, bool gpu);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, bool gpu);
protected:
	int _node_id;
	int _node_num;
	DistriNetwork *_node_nets;
	CrossNodeData *_node_datas;
public:
	DistriNetwork *_network_data;
	CrossNodeData *_data;
};

int run_node_cpu(DistriNetwork *network, CrossNodeData *cnd);
int run_node_gpu(DistriNetwork *network, CrossNodeData *cnd);

#define ASYNC

#endif /* MULTINODESIMULATOR_H */

