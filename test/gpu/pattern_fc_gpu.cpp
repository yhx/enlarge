#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
	{
		printf("Need 2 paras. For example\n FR 1%%: %s num_neuron pop_num\n", argv[0]);
		return 0;
	}
	
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    const size_t pop_num = atoi(argv[2]);

	const real run_time = 1.0;
	const real dt = 1e-4;
	Network c(dt);

    Population *g[N + 1];

    for (int i = 0; i < pop_num; ++i) {
        g[i] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
            0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
    }

    real *delay = getConstArray((real)1*dt, N * N);

    const real scale = 1e3;
    const real w = 3.0 * scale;

    real *weight = getConstArray((real)w / N / (pop_num - 1), N * N);
    enum SpikeType type = Inh;
    SpikeType *inh_con = getConstArray(type, N * N);

    for (int i = 0; i < pop_num; ++i) {
        for (int j = 0; j < pop_num; ++j) {
            if (i != j) {
                c.connect(g[i], g[j], weight, delay, NULL, N * N);
            }
        }
    }

    delArray(weight);

    SGSim mn(&c, dt);	// cpu
	mn.run(run_time);

    return 0;
}
