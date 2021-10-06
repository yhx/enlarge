/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef MULTINCCLSIMULATOR_H
#define MULTINCCLSIMULATOR_H

#include <string>

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"
#include "../interface/Simulator.h"

using std::string;

class MultiNCCLSimulator : public Simulator {
public:
	MultiNCCLSimulator(Network *network, real dt);
	MultiNCCLSimulator(const string &path, real dt);
	~MultiNCCLSimulator();

	using Simulator::run;

	int mpi_init(int *argc, char ***argv);
	int build_net(int num = 0, SplitType split=SynapseBalance, const char *name="", const AlgoPara *para = NULL);
	int save_net(const string &name);
	int load_net(const string &name);
	int distribute(SimInfo &, int);

	int run(real time, int gpu_num);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, int gpu_num);

public:
	DistriNetwork *_network_data;
	CrossNodeData *_data;

protected:
	int _proc_id;
	int _proc_num;
	int _gpu_id;
	int _gpu_num;
	int _gpu_grp;

	ncclComm_t comm_gpu;
	cudaStream_t _stream;

	DistriNetwork *_proc_nets;
	CrossNodeData *_proc_datas;
};

int run_proc_cpu(DistriNetwork *network, CrossMap *cnd, CrossSpike *msg);
int run_proc_gpu(DistriNetwork *network, CrossMap *cnd, CrossSpike *msg);

#define ASYNC

#endif /* MULTINCCLSIMULATOR_H */

