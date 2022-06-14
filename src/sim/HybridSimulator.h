#ifndef HYBRIDSIMULATOR_H
#define HYBRIDSIMULATOR_H

#include <string>

#include "../../msg_utils/msg_utils/HybridCrossMap.h"
#include "../../msg_utils/msg_utils/HybridProcBuf.h"
#include "../interface/Simulator.h"


// MPI_Comm local_comm;
pthread_barrier_t hybrid_thread_barrier;  // TODO: check whether it is right


class HybridSimulator : public Simulator {
public:
    HybridSimulator(Network *network, real dt);
    ~HybridSimulator();

    using Simulator::run;
    virtual int run(real time, FireInfo &log, int thread_num, int gpu_num, int k);

    int build_net(int num, SplitType split=SynapseBalance, const char *name="", const AlgoPara *para = NULL);
    int distribute(SimInfo &, int);

    DistriNetwork **_network_data;      // local network structure
	CrossNodeData **_data;              // shadow neuron information
protected:
    int _process_id;                    // current process id
    int _process_num;                   // total process number
    int _thread_id;                     // current thread id
    int _thread_num;                    // total thread number per process
    int _local_process_id;              // 局部进程id，为0则为放gpu线程
    int _gpu_num;                       // gpu的数量
    int _k;                             // 用于double buffer
    int _total_gpu_num;
    
    int _total_subnet_num;              // 子网络的总数
    int *_proc_gpu_num;                 // 每个进程控制的gpu数量
    int *_subnet_num;                   // 每个进程控制的子网络数量

    DistriNetwork *_all_nets;           // 划分的网络信息
	CrossNodeData *_all_datas;          // 每个线程发给其它线程的数据

    void run_cpu_hybrid(DistriNetwork *network, HybridCrossMap *cm, HybridProcBuf *pbuf, int run_thread_num, pthread_t *thread_ids);
};

/**
 * @brief 每一个线程的参数
 * 
 */
struct HybridRunPara {
	DistriNetwork *_net;
	HybridCrossMap *_cm;
	HybridProcBuf *_pbuf;
    int *_subnet_num;
    int _subnet_id;
    int _gpu_num;
    int _total_subnet_num;
    int _thread_id;
};

void *run_gpu_hybrid(void *para);
void *run_cpu_hybrid(void *para);

#endif
