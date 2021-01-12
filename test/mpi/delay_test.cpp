#include "../../include/BSim.h"
#include <stdlib.h>
#include<time.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	time_t start,end;
	start=clock(); //time(NULL);
	if(argc !=5)
	{
		printf("Need 4 paras, have %d. For example\n FR 1%%: %s depth num_neuron fire_rate delay\n", argc, argv[0]);
		return 0;
	}
	const int depth=atoi(argv[1]);
	const int N=atoi(argv[2]);

	const int fr = atoi(argv[3]);
	const int delay_step = atoi(argv[4]);
	real w1=0.0;
	real w2=0.0;
	real w3=0.0;
	real w4=0.0;
	int who=0;

	switch(fr) {
		case 5:
			w1 = 0.7;
			w2 = 0.9;
			w3 = 0.6;
			w4 = 0.2;
			who = 6;
			break;
		case 20:
			w1 = 1.3;
			w2 = 1;
			w3 = 2;
			w4 = 1;
			who = 50;
			break;
		default:
			w1 = 0.7;
			w2 = 0.5;
			w3 = 0.6;
			w4 = 0.3;
			who = 6;
	};

	// const real w1=atof(argv[3]);
	// const real w2=atof(argv[4]);
	// const real w3=atof(argv[5]);
	// const real w4=atof(argv[6]);
	// const int who=atoi(argv[7]);

	printf("depth=%d N=%d\n",depth,N);
	printf("w1=%f w2=%f w3=%f w4=%f who=%d\n", w1, w2, w3, w4, who);
	const real fthreshold=-54e-3;
	const real freset=-60e-3;
	const real c_m=0.2e-9; //*nF
	const real v_rest=-74e-3;//*mV
	const real tau_syn_e =2e-3;//*ms
	const real tau_syn_i= 0.6e-3;//*ms

	//# 论文没有这些参数
	const real tau_m=10e-3;//*ms
	const real i_offset =2e-9;//*nA
	const real frefractory=0;
	const real fv=-74e-3;

	const real run_time=1000e-3;
	const real dt=1e-4;

	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	Network c;

	if (node_id == 0) {

		Population *g[depth+1];

		g[1]=c.createPopulation(1, N, LIF_curr_exp(LIFNeuron(
						fv,v_rest,freset,
						c_m,tau_m,
						frefractory,tau_syn_e,tau_syn_i,
						fthreshold,i_offset
						),  tau_syn_e, tau_syn_i));

		for(int i=2;i<=depth;i++)
			g[i] = c.createPopulation(i, N, LIF_curr_exp(LIFNeuron(
							fv,v_rest,freset,
							c_m,tau_m,
							frefractory,tau_syn_e,tau_syn_i,
							fthreshold,0
							),  tau_syn_e, tau_syn_i));

		real * weight6 = getConstArray((real)(1e-9)*w1 /N, N*N);
		real * weight6_30 = getConstArray((real)(1e-9)*w2 /N, N*N);
		real * weight5 = getConstArray((real)(1e-9)*w3 /N, N*N);
		real * weight3 = getConstArray((real)(-1e-9)*w4 /N, N*N);
		// real * delay = getConstArray((real)0.4e-3, N*N);
		real * delay = getConstArray((real)(delay_step * 0.1e-3), N*N);

		enum SpikeType type=Inhibitory;
		SpikeType *ii = getConstArray(type, N*N);

		for(int i=2;i<=depth;i++)
		{

			c.connect(g[i-1], g[i], weight6, delay, NULL, N*N);
			if (i % who ==0)
				c.connect(g[1], g[i], weight6_30, delay, NULL, N*N);	
		}

		Population *p[depth+1];
		int i=1;
		while(i+1<=depth)
		{
			p[i] = c.createPopulation(i+depth, N, LIF_curr_exp(LIFNeuron(
							fv,v_rest,freset,
							c_m,tau_m,
							frefractory,tau_syn_e,tau_syn_i,
							fthreshold,0
							),  tau_syn_e, tau_syn_i));
			c.connect(g[i], p[i], weight5, delay, NULL, N*N);
			c.connect(p[i], g[i+1], weight3, delay, ii, N*N);
			i+=4;


		}
		//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
		
		delArray(weight6);
		delArray(weight6_30);
		delArray(weight5);
		delArray(weight3);
		delArray(delay);
		delArray(ii);
	}


#if 0
	STSim st(&c, dt);	// cpu
	st.run(run_time);
#else
	SGSim sg(&c, dt);	//gpu
	sg.run(run_time);	
#endif

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
