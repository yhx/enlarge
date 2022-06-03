#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	if (argc != 5)
	{
		printf("Need 4 paras. For example\n FR 1%%: %s num_neuron pop_num parts_num delay_num\n", argv[0]);
		return 0;
	}
	
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    const size_t pop_num = atoi(argv[2]);
    const int parts = atoi(argv[3]);  // 划分为几个节点运行程序
    const int delay_num = atoi(argv[4]);

	const real run_time = 1.0;
	const real dt = 1e-4;
	Network c(dt);

    if (node_id == 0) {
        Population *g[N + 1];

        const int diff_num = 5000;
        for (int i = 0; i < pop_num; ++i) {
            g[i] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        }

        real *delay = getConstArray((real)delay_num*dt, N * N);

        for (int i = 0; i < N * N; ++i) {
            delay[i] = ((i % delay_num) + 1) * dt;
        }

        const real scale = 1e3;
        const real w = 3.5 * scale;

        real *weight = getConstArray((real)w / N / (pop_num - 1), N * N);

        for (int i = 0; i < pop_num; ++i) {
            for (int j = 0; j < pop_num; ++j) {
                if (i != j) {
                    c.connect(g[i], g[j], weight, delay, NULL, N * N);
                }
            }
        }

        delArray(weight);
    }

    MNSim mn(&c, dt);	// cpu
	// mn.run(run_time, 1);

    if (node_id == 0) {
        // int parts = 16;
        SplitType split = SynapseBalance;
        string file_name = "pattern_fc_iaf_mpi_" + to_string(N) + "_" + to_string(pop_num);
        const char * name = file_name.c_str();
        mn.build_net(parts, split, name);
        mn.save_net(name);
    }

    return 0;
}
