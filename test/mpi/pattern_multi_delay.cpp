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

	if (argc != 2)
	{
		printf("Need 1 paras. For example\n FR 1%%: %s num_neuron\n", argv[0]);
		// printf("Need 7 paras. For example\n FR 1%%: %s depth num_neuron 0.7 0.5 0.6 0.3 6\n FR 5%%: %s depth num_neuron 0.7 0.9 0.6 0.2 6\n FR 20%%: %s depth num_neuron 1.3 1 2 1 50\n FR 50%%: %s depth num_neuron 2.7 3 2 1 20\n FR 65%%: %s depth num_neuron 4 2 5 3 20\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
		return 0;
	}
	
	const int depth = 3;  // 网络深度  
	const size_t N = atoi(argv[1]);  // 每层神经元的数量

	const real run_time = 1.0;
	const real dt = 1e-3;
	Network c(dt);

    if (node_id == 0) {
        Population *g[depth + 1];

        const int diff_num = 5000;
        g[1] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 37600.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        
        g[2] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        
        g[3] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));

        real *delay = getConstArray((real)dt, N * N);  
		for (int i = 0; i < N * N; ++i) {  // delay是dt, 2dt, ..., 10dt
			delay[i] = dt * (i % 10 + 1);
		}

        const real scale = 1e3;
        const real w1_2 = 2.4 * scale;
        const real w2_3 = 2.9 * scale;

        real *weight1_2 = getConstArray((real)w1_2 / N, N * N);
        real *weight2_3 = getConstArray((real)w2_3 / N, N * N);

        enum SpikeType type = Inh;
        SpikeType *inh_con = getConstArray(type, N * N);

        c.connect(g[1], g[2], weight1_2, delay, NULL, N * N);
        // for (int i = 0; i < N * N; ++i) {  // delay是0, dt, 2dt, ..., 9dt
		// 	delay[i] = dt * (i % 10 + 1) + dt;
		// }
        c.connect(g[2], g[3], weight2_3, delay, NULL, N * N);

        delArray(weight1_2);
        delArray(weight2_3);
        delArray(inh_con);
    }

    MNSim mn(&c, dt);	
	mn.run(run_time, 1);

    // if (node_id == 0) {
    //     int parts = 16;
    //     SplitType split = SynapseBalance;
    //     const char * name = "pattern_circle_iaf_mpi";
    //     mn.build_net(parts, split, name);
    //     mn.save_net(name);
    // }

    return 0;
}
