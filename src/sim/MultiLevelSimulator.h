/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef MULTILEVELSIMULATOR_H
#define MULTILEVELSIMULATOR_H

#include <string>

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/ProcBuf.h"
#include "../interface/Simulator.h"

using std::string;

class MultiLevelSimulator : public Simulator {
public:
	MultiLevelSimulator(Network *network, real dt);
	MultiLevelSimulator(const string &path, real dt, int thread_num);
	~MultiLevelSimulator();

	using Simulator::run;

	int mpi_init(int *argc, char ***argv);
	int build_net(int num, SplitType split=SynapseBalance, const char *name="", const AlgoPara *para = NULL);
	int save_net(const string &name);
	int load_net(const string &name);
	int distribute(SimInfo &, int);

	int run(real time, int thread_num);
	virtual int run(real time, int thread_num, bool gpu);
	virtual int run(real time, FireInfo &log);
	virtual int run(real time, FireInfo &log, int thread_num, bool gpu);

public:
	DistriNetwork **_network_data;
	CrossNodeData **_data;

protected:
	int _proc_id;
	int _proc_num;
	int _thread_num;

	DistriNetwork *_all_nets;
	CrossNodeData *_all_datas;
};

struct RunPara {
	DistriNetwork *_net;
	CrossMap *_cm;
	ProcBuf *_pbuf;
	int _thread_id;
};

// int run_proc_cpu(DistriNetwork *network, CrossMap *cnd, CrossSpike *msg);
// int run_proc_gpu(DistriNetwork **network, CrossMap **cnd, CrossSpike **msg);

extern pthread_barrier_t g_proc_barrier;

void * run_thread_ml(void *para);
void * run_gpu_ml(void *para);

#define ASYNC

#endif /* MULTILEVELSIMULATOR_H */

