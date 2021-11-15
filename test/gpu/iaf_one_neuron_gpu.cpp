#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	time_t start, end;
	start = clock(); //time(NULL);

	const int N = 1;  // 仅有一个神经元

	const real run_time = 1;
	const real dt = 1e-4;
	Network c(dt);

	Population *g[5];

    // IAFNeuron::IAFNeuron(real dt, size_t num, real Tau, real C, real t_ref, real E_L, real I_e,
	// 		real Theta, real V_reset, real tau_ex, real tau_in, real rho,
	// 		real delta, real V_m, real i_0, real i_1, real i_syn_in, real i_syn_ex)
	g[0] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2.0*dt, -70.0, 376.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
			0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));

	real *delay = getConstArray((real)1e-4, N);
	const real w_p = 0;
	real *weight_p = getConstArray((real) w_p / N, N);

	real poisson_mean = 40;
	real *p_m = getConstArray(poisson_mean, N);

    c.connect_poisson_generator(g[0], p_m, weight_p, delay, NULL);

	c.log_graph();

	cout << "finish build network!" << endl;
	cout << "node number: " << c._node_num << endl;

	delArray(weight_p);

	SGSim st(&c, dt);	// cpu
	st.run(run_time);

	end = clock(); //time(NULL);
	printf("exec time=%lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
