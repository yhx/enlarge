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
	
	const int N = 10;
    const real dt = 0.5e-4;
    const real run_time = 1000e-3;
	Network c(dt);

    if (node_id == 0) {
        // parameter for LIF model
        const real fthreshold = -54e-3;
        const real freset = -60e-3;
        const real c_m = 0.2e-9;	   //*nF
        const real v_rest = -74e-3;	   //*mV
        const real tau_syn_e = 2e-3;   //*ms
        const real tau_syn_i = 0.6e-3; //*ms

        //# 论文没有这些参数
        const real tau_m = 10e-3;	//*ms
        const real i_offset = 4.6e-10; //*nA
        const real i_offset2 = 4.6e-10; //*nA
        const real frefractory = 0;
        const real fv = -74e-3;

        Population *g[5];

        const int diff_num = 5000;

        g[0] = c.createPopulation(1, N, LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e, tau_syn_i, fthreshold, 0, dt));
        g[1] = c.createPopulation(1, N, LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e, tau_syn_i, fthreshold, 0, dt));

        real *delay = getConstArray((real)1e-4, N * N);

        const real w = 2.4;
        real *weight = getConstArray((real)(1e-9) * w / N, N * N);

        enum SpikeType type = Inh;
        SpikeType *inh_con = getConstArray(type, N * N);
        real poisson_mean = 4;
        real *p_m = getConstArray(poisson_mean, N);

        c.connect_poisson_generator(g[0], p_m, weight, delay, NULL);
        c.connect(g[0], g[1], weight, delay, NULL, N * N);

        delArray(weight);
	    delArray(inh_con);
    }


	MNSim mn(&c, dt); //gpu
	mn.run(run_time);

	return 0;
}
