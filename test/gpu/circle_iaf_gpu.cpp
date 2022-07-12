#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	time_t start, end;
	start = clock(); //time(NULL);
	if (argc != 2)
	{
		printf("Need 1 paras. For example\n FR 1%%: %s depth num_neuron\n", argv[0]);
		// printf("Need 7 paras. For example\n FR 1%%: %s depth num_neuron 0.7 0.5 0.6 0.3 6\n FR 5%%: %s depth num_neuron 0.7 0.9 0.6 0.2 6\n FR 20%%: %s depth num_neuron 1.3 1 2 1 50\n FR 50%%: %s depth num_neuron 2.7 3 2 1 20\n FR 65%%: %s depth num_neuron 4 2 5 3 20\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
		return 0;
	}
	
	const int depth = 3;  // 网络深度  
	const int N = atoi(argv[1]);  // 每层神经元的数量

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

	Population *g[depth + 1];

    const int diff_num = 5000;
	// input
	g[1] = c.createPopulation(1, N, LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e, tau_syn_i, fthreshold, i_offset, dt));
    g[2] = c.createPopulation(2, N, LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e, tau_syn_i, fthreshold, 0, dt));
    g[3] = c.createPopulation(3, N, LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e, tau_syn_i, fthreshold, 0, dt));

	real *delay = getConstArray((real)1e-4, N * N);

	const real w1_2 = 2.4;
	const real w2_3 = 2.9;
	const real w3_1 = -1.0;  // inh

	real *weight1_2 = getConstArray((real)(1e-9) * w1_2 / N, N * N);
	real *weight2_3 = getConstArray((real)(1e-9) * w2_3 / N, N * N);
	real *weight3_1 = getConstArray((real)(1e-9) * w3_1 / N, N * N);

	enum SpikeType type = Inh;
	SpikeType *inh_con = getConstArray(type, N * N);

    c.connect(g[1], g[2], weight1_2, delay, NULL, N * N);
    c.connect(g[2], g[3], weight2_3, delay, NULL, N * N);
    c.connect(g[3], g[1], weight3_1, delay, inh_con, N * N);

	c.log_graph();

	cout << "finish build network!" << endl;
	cout << "node number: " << c._node_num << endl;

	delArray(weight1_2);
	delArray(weight2_3);
	delArray(weight3_1);
	delArray(inh_con);

#if 0
	STSim st(&c, dt);	// cpu
	st.run(run_time);
#else
	SGSim gs(&c, dt); //gpu
	gs.run(run_time);
#endif

	end = clock(); //time(NULL);
	printf("exec time=%lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return 0;
}
