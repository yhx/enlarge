#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	time_t start, end;
	start = clock(); //time(NULL);
	// if (argc != 3)
	// {
	// 	printf("Need 2 paras. For example\n FR 1%%: %s depth num_neuron\n", argv[0]);
	// 	// printf("Need 7 paras. For example\n FR 1%%: %s depth num_neuron 0.7 0.5 0.6 0.3 6\n FR 5%%: %s depth num_neuron 0.7 0.9 0.6 0.2 6\n FR 20%%: %s depth num_neuron 1.3 1 2 1 50\n FR 50%%: %s depth num_neuron 2.7 3 2 1 20\n FR 65%%: %s depth num_neuron 4 2 5 3 20\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
	// 	return 0;
	// }
	
	// const int depth = atoi(argv[1]);  // 网络深度  
	// const int N = atoi(argv[2]);  // 每层神经元的数量
	const int N = 10;

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

	const real run_time = 1000e-3;
	const real dt = 0.5e-4;
	Network c(dt);

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

	c.log_graph();

	cout << "finish build network!" << endl;
	cout << "node number: " << c._node_num << endl;

	delArray(weight);
	delArray(inh_con);

	STSim st(&c, dt);	// cpu
	st.run(run_time);

	end = clock(); //time(NULL);
	printf("exec time=%lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return 0;
}

