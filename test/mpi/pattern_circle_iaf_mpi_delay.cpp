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

	if (argc != 4)
	{
		printf("Need 3 paras. For example\n FR 1%%: %s num_neuron parts_num delay_num\n", argv[0]);
		// printf("Need 7 paras. For example\n FR 1%%: %s depth num_neuron 0.7 0.5 0.6 0.3 6\n FR 5%%: %s depth num_neuron 0.7 0.9 0.6 0.2 6\n FR 20%%: %s depth num_neuron 1.3 1 2 1 50\n FR 50%%: %s depth num_neuron 2.7 3 2 1 20\n FR 65%%: %s depth num_neuron 4 2 5 3 20\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
		return 0;
	}
	
	const int depth = 3;  // 网络深度  
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    const int parts = atoi(argv[2]);  // 划分为几个节点运行程序
    const int delay_num = atoi(argv[3]);

	const real run_time = 1.0;
	const real dt = 1e-4;
	Network c(dt);

    if (node_id == 0) {
        Population *g[depth + 1];

        const int diff_num = 5000;
        g[1] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        
        g[2] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        
        g[3] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));

        real *delay = getConstArray((real)delay_num*dt, N * N);

        const real scale = 1e3;
        const real w1_2 = 2.4 * scale;
        const real w2_3 = 2.9 * scale;
        const real w3_1 = -1.0 * scale;

        real *weight1_2 = getConstArray((real)w1_2 / N, N * N);
        real *weight2_3 = getConstArray((real)w2_3 / N, N * N);
        real *weight3_1 = getConstArray((real)w3_1 / N, N * N);

        enum SpikeType type = Inh;
        SpikeType *inh_con = getConstArray(type, N * N);

        c.connect(g[1], g[2], weight1_2, delay, NULL, N * N);
        c.connect(g[2], g[3], weight2_3, delay, NULL, N * N);
        c.connect(g[3], g[1], weight3_1, delay, inh_con, N * N);

        delArray(weight1_2);
        delArray(weight2_3);
        delArray(weight3_1);
        delArray(inh_con);
    }

    MNSim mn(&c, dt);	// cpu
	// mn.run(run_time, 0);

    if (node_id == 0) {
        // int parts = 16;
        SplitType split = SynapseBalance;
        string file_name = "pattern_circle_iaf_mpi_" + to_string(N);
        const char * name = file_name.c_str();
        mn.build_net(parts, split, name);
        mn.save_net(name);
    }

    return 0;
}
