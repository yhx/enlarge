#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <string>
#include <mpi.h>

#include "../utils/utils.h"
#include "../base/TypeFunc.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../msg_utils/msg_utils/msg_utils.h"
#include "../msg_utils/convert.h"
#include "../net/Network.h"
#include "../neuron/lif/LIFData.h"
#include "../neuron/iaf/IAFData.h"
#include "HybridSimulator.h"

using std::string;
using std::to_string;


/**
 * @brief Construct a new Hybrid Simulator:: Hybrid Simulator object
 * 并将进程域划分为node_num个子域用于确定哪个进程属于控制GPU的进程，哪个进程是纯CPU的进程
 * @param network 
 * @param dt 
 */
HybridSimulator::HybridSimulator(Network *network, real dt) : Simulator(network, dt) {
    _all_nets = NULL;
    _all_datas = NULL;
    _network_data = NULL;
    _data = NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &_process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &_process_num);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);  // 获得本进程的机器名
    printf("Proccessor %s, rank %d out of %d processors\n", processor_name, _process_id, _process_num);
    _thread_num = 0;
    _proc_gpu_num = (int *)malloc_c<int>(_process_num);
    _subnet_num = (int *)malloc_c<int>(_process_num);
    
    /**
     * @brief 将通信域划分为node_num个子域，用于确定CPU进程和GPU进程
     */
    char *all_processor_name = (char *)malloc_c<char>(_process_num * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

    int count = 0, cur_node_gpu_num;
    for (int i = 0; i < _process_num; ++i) {
        if (strcmp(processor_name, all_processor_name + MPI_MAX_PROCESSOR_NAME * i) == 0) {
            if (_process_id == i) {
                if (count == 0) {
                    cur_node_gpu_num = -1;  // control gpu
                }
                else {
                    cur_node_gpu_num = 0;   // only control cpu
                }
            }
            count++;
        }
    }
    MPI_Allgather(&cur_node_gpu_num, 1, MPI_INT, _proc_gpu_num, 1, MPI_INT, MPI_COMM_WORLD);
}


/**
 * @brief 搭建完整的网络
 * 
 * @param num 总线程数 _process_num * _thread_num
 * @param split 划分类型
 * @param name 网络名称
 * @param para 划分算法参数
 * @return int 
 */
int HybridSimulator::build_net(int num, SplitType split, const char *name, const AlgoPara *para) {
	SimInfo info(_dt);
	info.save_mem = true;

	if (!_all_nets) {  // 如果当前没有创建完整网络
		assert(num > 1);  // 线程总数应大于1

		_network->set_node_num(num);  // 将网络分为线程总数份
        // _all_nets类型为DistriNetwork*
        // buildNetworks会划分网络结构
		_all_nets = _network->buildNetworks(info, split, name, para);  
		for (int i = 0; i < num; i++) {
			_all_nets[i]._simCycle = 0;
			_all_nets[i]._nodeIdx = i;
			_all_nets[i]._nodeNum = num;
			_all_nets[i]._dt = _dt;
		}
	}

	if (!_all_datas) {  // 为_all_datas分配空间
		_all_datas = _network->arrangeCrossNodeData(info);
	}

	return 0;
}


/**
 * @brief 在进程0中构建网络、划分并将其它进程的DistriNetwork和CrossNodeData数据发送出去，
 * 而其余进程接受进程0发送的DistriNetwork和CrossNodeData数据。
 * 
 * @param info 仿真参数
 * @param sim_cycle 仿真的step数
 * @return int 
 */
int HybridSimulator::distribute(SimInfo &info, int sim_cycle)
{
	printf("Distributing Network\n");
    int total_gpu_num = 0;
    for (int i = 0; i < _process_num; ++i) {
        total_gpu_num += _proc_gpu_num[i];  // control gpu
    }
	int part_num = _process_num + _k * total_gpu_num;  // 默认分配方式：每个GPU对应k_个网络、每个CPU对应一个网络
	if (_process_id == 0) {  // 只在进程1分配全局的网络信息
		// print_mem("Finish Network");
		// print_mem("Finish CND");
		if (!(_all_nets && _all_datas)) {
			build_net(part_num);  // 将整个网络划分为num个子网络
		}

		for (int i = 0; i < part_num; i++) {
			_all_nets[i]._simCycle = sim_cycle;
			_all_nets[i]._nodeIdx = i;
			_all_nets[i]._nodeNum = part_num;
			_all_nets[i]._dt = _dt;
		}
	} else {  // 其它进程等待第一个进程完成网络划分
		printf("Rank %d, Wait for network build\n", _process_id);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	_network_data = malloc_c<DistriNetwork *>(part_num);  // 创建数组DistriNetwork[part_num]
	_data = malloc_c<CrossNodeData *>(part_num);  // 创建数组CrossNodeData[part_num]

	if (_process_id == 0) {  // 在编号为0的进程为每个进程中的每个线程分配对应的子网络 
        
		for (int i = 0; i < part_num; i++) {  // 为每个part分配网络和参数数据
			_network_data[i] = &(_all_nets[i]);
			_data[i] = &(_all_datas[i]);
		}

        int _net_id_start[_process_num];
        _net_id_start[0] = 0;
        for (int i = 1; i < _process_num; ++i) {
            _net_id_start[i] = _net_id_start[i - 1] + _subnet_num[i - 1];
        }

        // 进程0为编号为1~_process_num-1的进程发送对应的DistriNetwork和CrossNodeData
		for (int p = 1; p < _process_num; p++) {
			int tag = DATA_TAG;
            /**
             * @brief 
             * process 0 需要向 process p 发送的子网络数据编号为：
             * _net_id_start[p] 到 _net_id_start[p] + _proc_gpu_num[p] * _k + 1
             */
			for (int i = _net_id_start[p]; i < _net_id_start[p] +_subnet_num[p]; i++) {
				int dst = p;
				printf("Send to %d, tag: %d\n", dst, tag);
				sendDistriNet(&(_all_nets[i]), dst, tag, MPI_COMM_WORLD);
				printf("Send DistriNet to %d, tag: %d\n", dst, tag);
				tag += DNET_TAG;
				sendCND(&(_all_datas[i]), dst, tag, MPI_COMM_WORLD);
				printf("Send CND to %d, tag: %d\n", dst, tag);
				tag += CND_TAG;
			}
		}
	} else {
        // 编号非0的进程从编号为0的进程接受DistriNetwork和CrossNodeData的数据
		int tag = DATA_TAG;
		for (int t = 0; t < _subnet_num[_process_id]; t++) {
			printf("%d recv from %d, tag: %d\n", _process_id, 0, tag);
			_network_data[t] = recvDistriNet(0, tag, MPI_COMM_WORLD);
			printf("%d recv DistriNet from %d, tag: %d\n", _process_id, 0, tag);
			tag += DNET_TAG;
			_data[t] = recvCND(0, tag, MPI_COMM_WORLD);
			printf("%d recv CND from %d, tag: %d\n", _process_id, 0, tag);
			tag += CND_TAG;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
}


/**
 * @brief 每个process的整体控制
 * 
 * @param time 总仿真时间
 * @param log 存储信息
 * @param thread_num 每个进程的最大线程数量（应大于gpu_num）
 * @param gpu_num 每个节点的gpu数量
 * @return int 0
 */
int HybridSimulator::run(real time, FireInfo &log, int thread_num, int gpu_num, int k) {
    assert(thread_num > gpu_num);  // 线程数应大于受线程控制的gpu的数目

    /**
     * 计算得到每个进程所控制的gpu数量
     */
    _total_subnet_num = 0;
    for (int i = 0; i < _process_num; ++i) {
        if (_proc_gpu_num[i] == -1) {
            _proc_gpu_num[i] = gpu_num;
            _subnet_num[i] = gpu_num * k + 1;
        } 
        else {
            _proc_gpu_num[i] = 0;
            _subnet_num[i] = 1;
        }
        _total_subnet_num += _subnet_num[i];
    }
    
    _total_gpu_num = 0;
    for (int i = 0; i < _process_num; ++i) {
        _total_gpu_num += _proc_gpu_num[i];  // control gpu
    }

    printf("TOTAL GPU NUM: %d\n", _total_gpu_num);
    MPI_Barrier(MPI_COMM_WORLD);
    
    _thread_num = thread_num;   // 每个进程可分配的线程数
    _gpu_num = gpu_num;         // 每个机器上的gpu的数量
    _k = k;                     // 用于double buffer，一个GPU控制k个进程

    // total simulation steps
    int sim_cycle = round(time / _dt);

    // log info
    SimInfo info(_dt);
    info.save_mem = true;

    to_attach();  // used to debug MPI

    printf("Rank %d, before distributed\n", _process_id);

    if (!(_network_data && _data)) {  // 划分网络
        distribute(info, sim_cycle);
    }

    // TODO: 检查cm和cs是否正确
    // 当前节点需要分配的网络个数为1 + k * _proc_gpu_num[_process_id]个
    HybridCrossMap **cm = malloc_c<HybridCrossMap*>(_subnet_num[_process_id]);       // 和实例的数目相同，主要用来映射，本地id变到对面的id是什么样的
    HybridCrossSpike **cs = malloc_c<HybridCrossSpike*>(_subnet_num[_process_id]);   // procbuf的结构需要重新写

    // 转换数据结构，将DistriNetwork和CrossNodeData转化为HybridCrossMap和HybridCrossSpike
    int start_subnet_id = 0;
    for (int i = 0; i < _process_id - 1; ++i) {
        start_subnet_id += _subnet_num[i];
    }
    for (int i = 0; i < _subnet_num[_process_id]; ++i) {
        if (_network_data[i]->_simCycle != sim_cycle) {
            _network_data[i]->_simCycle = sim_cycle;
        }
        cm[i] = convert2hybridcrossmap(_network_data[i]->_crossnodeMap);
        cs[i] = convert2hybridcrossspike(_data[i], start_subnet_id + i, 0);
    }

    /** 
     * 当前节点的线程数为_proc_gpu_num[_process_id] + 1:
     * 如果当前节点为gpu节点，则线程总数为gpu数+1;
     * 否则为1;
     **/
    assert(_thread_num > 0);  
    pthread_barrier_init(&hybrid_thread_barrier, NULL, _thread_num);
    pthread_t *thread_ids = malloc_c<pthread_t>(_thread_num);
    assert(thread_ids != NULL);

    // TODO: 修改pbuf的大小，以备后续调用
    HybridProcBuf pbuf(cs, &hybrid_thread_barrier, _process_id, _process_num, cs[0]->_min_delay, _subnet_num, _k, _total_subnet_num, _thread_num);

    HybridRunPara *paras = malloc_c<HybridRunPara>(_subnet_num[_process_id]);  // 为每一个线程分配一个RunPara

    /**
     * 分配线程。如果当前进程不控制gpu，则不需要分配线程，在调用的子程序中分配线程即可
     * 而对于控制gpu的进程，则需要分配线程来管理
     **/
    printf("Rank %d, launch thread\n", _process_id);
    for (int i = 0; i < _subnet_num[_process_id]; ++i) {
        paras[i]._net = _network_data[i];
        paras[i]._cm = cm[i];
        paras[i]._pbuf = &pbuf;
        paras[i]._subnet_id = i;
        paras[i]._gpu_num = _gpu_num;
        paras[i]._subnet_num = _subnet_num;
        paras[i]._total_subnet_num = _total_subnet_num;
        paras[i]._thread_id = i % _proc_gpu_num[_process_id];  // TODO：在double buffer情况下需要改正这一个值

        if (i < _proc_gpu_num[_process_id]) {  // 如果当前的线程是GPU进程（_proc_gpu_num[_process_id]>0），前gpu_num个线程每个对应一个gpu
            int ret = pthread_create(&(thread_ids[i]), NULL, &run_gpu_hybrid, (void*)&(paras[i]));
            assert(ret == 0);
        } else {  // 否则调用管理cpu线程的函数
            // paras[i]._run_thread_num = thread_num - _proc_gpu_num[_process_id];
            run_cpu_hybrid(_network_data[i], cm[i], &pbuf, thread_num - _proc_gpu_num[_process_id], thread_ids, i);
        }
    }

    for (int i = 0; i < _proc_gpu_num[_process_id] + 1; ++i) {
        pthread_join(thread_ids[i], NULL);
    }

    pthread_barrier_destroy(&hybrid_thread_barrier);

    free_c(paras);
 
    for (int i = 0 ; i < _thread_num; ++i) {
        delete cm[i];
        delete cs[i];
    }

    free_c(cm);
    free_c(cs);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}

/**
 * @brief data structure for pthread parameters used in multi-thread CPU simulation
 */
struct HybridCPUPthreadPara {
    DistriNetwork *_network;
	Buffer *_buffer;
	size_t _thread_num;
	size_t _thread_id;
    HybridProcBuf *pbuf;
    size_t _cpu_control_thread_id;
    size_t _subnet_id;
};

void HybridSimulator::run_cpu_hybrid(DistriNetwork *network, HybridCrossMap *cm, HybridProcBuf *pbuf, int run_thread_num, pthread_t *thread_ids, int subnet_id) {
    HybridCPUPthreadPara *paras = malloc_c<HybridCPUPthreadPara>(run_thread_num);
    
    GNetwork *pNetCPU = network->_network;
    int nTypeNum = pNetCPU->nTypeNum;
    int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
    int max_delay = pNetCPU->ppConnections[0]->maxDelay;
    Buffer buffer(pNetCPU->bufferOffsets[nTypeNum], allNeuronNum, max_delay);

    for (size_t i = 0; i < _thread_num - _proc_gpu_num[_process_id]; ++i) {
        paras[i]._network = network;
	    paras[i]._buffer = &buffer;
	    paras[i]._thread_num = run_thread_num;
	    paras[i]._thread_id = i;
        paras[i].pbuf = pbuf;
        paras[i]._cpu_control_thread_id = thread_ids[_proc_gpu_num[_process_id]];
        paras[i]._subnet_id = subnet_id;
        int ret = pthread_create(&(thread_ids[i + _proc_gpu_num[_process_id]]), NULL, hybrid_sim_multi_thread_cpu, (void*)&(paras[i]));
    }

    for (int i = 0; i < run_thread_num; ++i) {
        pthread_join(thread_ids[i], NULL);
    }

    free_c(paras);
}

void *hybrid_sim_multi_thread_cpu(void *paras) {
	HybridCPUPthreadPara *tmp = static_cast<HybridCPUPthreadPara*>(paras);
    DistriNetwork *network = tmp->_network;
    GNetwork *pNetCPU = pNetCPU = network->_network;
    Buffer *buffer = tmp->_buffer;
    size_t thread_id = tmp->_thread_id;
    size_t thread_num = tmp->_thread_num;
    HybridProcBuf *pbuf = tmp->pbuf;
    size_t cpu_control_thread_id = tmp->_cpu_control_thread_id;
    size_t subnet_id = tmp->_subnet_id;

    int nTypeNum = pNetCPU->nTypeNum;
	int sTypeNum = pNetCPU->sTypeNum;
    int allNeuronNum = pNetCPU->ppConnections[0]->nNum;
    int max_delay = pNetCPU->ppConnections[0]->maxDelay;
	int min_delay = pNetCPU->ppConnections[0]->minDelay;

#ifdef PROF
    FILE *v_file = NULL, *sim_file = NULL;
	if (thread_id == 0) {
        v_file = log_file_mpi("v", network->_nodeIdx);
	    sim_file = log_file_mpi("sim", network->_nodeIdx);
		printf("Start runing for %d cycles\n", network->_simCycle);
	}
#endif

	struct timeval ts, te;
	gettimeofday(&ts, NULL);

	for (int time = 0; time < network->_simCycle; time++) {
		int currentIdx = time % (max_delay + 1);
		buffer->_fired_sizes[currentIdx] = 0;

		for (int i = 0; i < nTypeNum; i++) {
			updatePthreadType[pNetCPU->pNTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppNeurons[i],
			 	buffer->_data + pNetCPU->bufferOffsets[i], buffer->_fire_table, buffer->_fired_sizes, 
				buffer->_fire_table_cap, pNetCPU->pNeuronNums[i+1]-pNetCPU->pNeuronNums[i],
				pNetCPU->pNeuronNums[i], time, thread_num, thread_id - cpu_control_thread_id, hybrid_thread_barrier);
		}
        // 同步1，一次通信
        pbuf->fetch_cpu();

        if (thread_id == cpu_control_thread_id) {
            pbuf->update_cpu(thread_id, subnet_id, time);
        }

		for (int i = 0; i < sTypeNum; i++) {
			updatePthreadType[pNetCPU->pSTypes[i]](pNetCPU->ppConnections[i], pNetCPU->ppSynapses[i], 
				buffer->_data, buffer->_fire_table, buffer->_fired_sizes, buffer->_fire_table_cap,
				pNetCPU->pSynapseNums[i+1] - pNetCPU->pSynapseNums[i], pNetCPU->pSynapseNums[i], time, thread_num,
				thread_id - cpu_control_thread_id, hybrid_thread_barrier);
		}
        
        if (thread_id == cpu_control_thread_id) {
            pbuf->upload_cpu(thread_id, subnet_id, buffer->_fire_table, buffer->_fired_sizes, buffer->_fired_sizes, buffer->_fire_table_cap, max_delay, time); 
        }
	}

	gettimeofday(&te, NULL);

	double seconds =  te.tv_sec - ts.tv_sec + (te.tv_usec - ts.tv_usec)/1000000.0;
	printf("Thread %d Simulation finesed in %lfs\n", network->_nodeIdx, seconds);

#ifdef PROF
	if (thread_id == 0) {
        char name[512];
	    sprintf(name, "mpi_%d", network->_nodeIdx); 
		for (int i = 0; i < nTypeNum; i++) {
			logRateNeuron[pNetCPU->pNTypes[i]](pNetCPU->ppNeurons[i], "cpu");
		}

		fclose(v_file);
		fclose(sim_file);
	}
#endif

	return 0;
}

