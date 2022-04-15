#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main(int argc, char **argv)
{
    srand(std::time(0));
	time_t start, end;
	start = clock(); //time(NULL);

	const int N = 1;

	const real run_time = 1000e-3;
	const real dt = 1e-4;
	Network c(dt);

	Population *g[5];

    const int diff_num = 5000;

	g[0] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2.0*dt, -70.0, 376.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
			0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
	g[1] = c.createPopulation(2, N, IAFNeuron(dt, 1, 10.0, 250.0, 2.0*dt, -70.0, 0.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
			0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));

	real *delay = getConstArray((real)1e-4, N * N);

	const real w_p = 0.2;
	real *weight_p = getConstArray((real) w_p / N, N);

	enum SpikeType type = Inh;
	SpikeType *inh_con = getConstArray(type, N * N);
	real poisson_mean = 15000 * dt;
	real *p_m = getConstArray(poisson_mean, N);

    c.connect_poisson_generator(g[0], p_m, weight_p, delay, NULL);
    c.connect_poisson_generator(g[1], p_m, weight_p, delay, NULL);

	c.log_graph();

	cout << "finish build network!" << endl;
	cout << "node number: " << c._node_num << endl;

	delArray(weight_p);
	delArray(inh_con);

	STSim st(&c, dt);	// cpu
	st.run(run_time);

	end = clock(); //time(NULL);
	printf("exec time=%lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
