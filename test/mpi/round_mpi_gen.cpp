
#include "../../include/BSim.h"
#include <stdlib.h>
#include<time.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	time_t start,end;
	start=clock(); //time(NULL);
	if(argc !=6)
	{
		if (node_id == 0) {
			printf("Need 5 paras. For example\n FR 1%%: %s n0 n1 fire_rate delay net_parts\n", argv[0]);
		}
		return 0;
	}
	const size_t n0=(size_t)atoi(argv[1]);
	const size_t n1=(size_t)atoi(argv[2]);

	const int fr = atoi(argv[3]);
	const int delay_step = atoi(argv[4]);

	const int parts = atoi(argv[5]);

	real w1=0.0;
	real w2=0.0;

	switch(fr) {
		case 5:
			w1 = 0.7;
			w2 = -0.9;
			break;
		default:
			w1 = 2.0;
			w2 = -1.5;
	};

	const real i_offset =5.5e-10;//*nA

	if (node_id == 0) {
		printf("n0=%ld n1=%ld\n", n0, n1);
		printf("w1=%f w2=%f\n", w1, w2);
	}

	const real fthreshold=-54e-3;
	const real freset=-60e-3;
	const real c_m=0.2e-9; //*nF
	const real v_rest=-74e-3;//*mV
	const real tau_syn_e =2e-3;//*ms
	const real tau_syn_i= 0.6e-3;//*ms

	//# 论文没有这些参数
	const real tau_m=10e-3;//*ms
	const real frefractory=0;
	const real fv=-74e-3;

	// const real run_time=1000e-3;
	const real dt=1e-4;

	Network c(dt);

	if (node_id == 0) {

		Population *p0 = c.createPopulation(0, n0, 
				LIFNeuron(
						fv,
						v_rest,
						freset, 
						c_m,
						tau_m, 
						frefractory,
						tau_syn_e,tau_syn_i, 
						fthreshold,
						i_offset, dt));  

		Population *p1 = c.createPopulation(0, n1, 
				LIFNeuron(
						fv,
						v_rest,
						freset, 
						c_m,
						tau_m, 
						frefractory,
						tau_syn_e,tau_syn_i, 
						fthreshold,
						0, dt));  
		print_mem("After build neuron");

		real * weight01 = getConstArray((real)(1e-9)*w1/n0, n0*n1);
		real * weight10 = getConstArray((real)(1e-9)*w2/n1, n0*n1);
		real * delay = getConstArray((real)(delay_step * 0.1e-3), n0*n1);

		enum SpikeType type=Inh;
		SpikeType *ii = getConstArray(type, n0*n1);

		c.connect(p0, p1, weight01, delay, NULL, n0*n1);
		c.connect(p1, p0, weight10, delay, ii, n0*n1);

		print_mem("After build synapse");

		delArray(weight01);
		delArray(weight10);
		delArray(delay);
		delArray(ii);

		print_mem("After delete array");
	}

	MNSim mn(&c, dt);	//gpu
	mn.build_net(parts);
	mn.save_net("round_mpi_4_39999_39999_1_8");
	// mn.run(run_time);	

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
