#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	if (argc != 5)
	{
		printf("Need 3 paras. For example\n FR 1%%: %s num_neuron pop_num parts_num\n", argv[0]);
		return 0;
	}
	
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    const size_t pop_num = atoi(argv[2]);
    const int parts = atoi(argv[3]);  // 划分为几个节点运行程序

	const real run_time = 1.0;
	const real dt = 1e-4;
	Network c(dt);

    const int st = atoi(argv[4]);
    SplitType split = SynapseBalance;
    if(st == 0)
    {
        split = SynapseBalance;
    }
    else if(st == 1)
    {
        split = NeuronBalance;
    }
    else if(st == 2)
    {
        split = RoundRobin;
    }
    else{
        split = Metis;
    }

    if (node_id == 0) {
        srand(666);  // 初始化随机数种子保证每次结果一致 
        const real connect_prob = 0.05;
        const real inh_prob = 0.2;

        Population *g[N + 1];

        const int diff_num = 5000;
        for (int i = 0; i < pop_num; ++i) {
            g[i] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        }

        const real scale = 1e3;
        const real w = 3.5 * scale;

        vector<int> idx_src_;
        vector<int> idx_dst_;

        for (int i = 0; i < pop_num; ++i) {
            for (int j = 0; j < pop_num; ++j) {
                if (i != j) {
                    int syn_num = 0;
                    idx_src_.clear();
                    idx_dst_.clear();

                    for (int k = 0; k < N; ++k) {
                        for (int l = 0; l < N; ++l) {
                            // cout << rand() << endl;
                            if ( (real)(rand() % 10000) / 10000 < connect_prob) {
                                // cout << k << " " << l << endl;
                                idx_src_.push_back(k);
                                idx_dst_.push_back(l);
                                syn_num++;
                            }
                        }
                        // idx_weight_.push_back(w / count)
                    }

                    int *idx_src = new int[syn_num+1];              // 源神经元id数组
                    int *idx_dst = new int[syn_num+1];              // 目的神经元id数组
                    real* weight = getConstArray((real)w / syn_num, syn_num);
                    real *delay = getConstArray((real)2*dt, syn_num);
                    SpikeType *sp = getConstArray(Exc, syn_num);

                    for (int k = 0; k < idx_src_.size(); ++k) {
                        idx_src[k] = idx_src_[k];
                        idx_dst[k] = idx_dst_[k];
                        if ( (real)(rand() % 10000) / 10000 < inh_prob ) {
                            sp[k] = Inh;
                        }
                        real rand_num = (real)(rand() % 10000) / 10000 + 0.5;
                        weight[i] = (real) w / syn_num * rand_num * 10;
                    }

                    // c.connect(g[i], g[j], weight, delay, NULL, N * N);
                    c.connect(g[i], g[j], idx_src, idx_dst, syn_num, weight, delay, sp);

                    delArray(idx_src);
                    delArray(idx_dst);
                    delArray(weight);
                    delArray(delay);
                    delArray(sp);
                }
            }
        }

    }

    MNSim mn(&c, dt);	// cpu
	// mn.run(run_time, 1);

    if (node_id == 0) {
        // int parts = 16;
        // SplitType split = SynapseBalance;
        string file_name = "pattern_random_iaf_mpi_" + to_string(N) + "_" + to_string(pop_num);
        const char * name = file_name.c_str();
        mn.build_net(parts, split, name);
        mn.save_net(name);
    }

    return 0;
}
