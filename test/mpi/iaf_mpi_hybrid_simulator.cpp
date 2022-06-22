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

	const int N = 20;
	const real run_time = 1000e-3;
	const real dt = 1e-4;
	Network c(dt);

    if (node_id == 0) {
        Population *g[5];
        
        g[0] = c.createPopulation(1, N, IAFNeuron(dt));
        g[1] = c.createPopulation(2, N, IAFNeuron(dt));

        real *delay = getConstArray((real)1e-4, N * N);
        
        const real w = 2.4;
        const real w_p = 10;
        real *weight = getConstArray((real) w / N, N * N);
        real *weight_p = getConstArray((real) w_p / N, N);

        enum SpikeType type = Inh;
        SpikeType *inh_con = getConstArray(type, N * N);
        real poisson_mean = 40;
        real *p_m = getConstArray(poisson_mean, N);

        // c.connect_poisson_generator(g[0], p_m, weight_p, delay, NULL);
        c.connect(g[0], g[1], weight, delay, NULL, N * N);

        delArray(weight);
        delArray(weight_p);
        delArray(inh_con);
    }

	HSim hm(&c, dt);	// cpu
	hm.run(run_time, 28, 2, 1);

	return 0;
}
