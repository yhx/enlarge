#include <stdlib.h>
#include <time.h>

#include "../../include/BSim.h"

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	time_t start,end;
	start=clock(); //time(NULL);
	if(argc !=6 && argc != 7 && argc != 11)
	{
		printf("Need 5/6/10 paras. For example\n FR 1%%: %s depth num_neuron fire_rate delay net_parts [algorithm syn_weight comm_weight send_weight recv_weight]\n", argv[0]);
		// printf("Need 7 paras. For example\n FR 1%%: %s depth num_neuron 0.7 0.5 0.6 0.3 6\n FR 5%%: %s depth num_neuron 0.7 0.9 0.6 0.2 6\n FR 20%%: %s depth num_neuron 1.3 1 2 1 50\n FR 50%%: %s depth num_neuron 2.7 3 2 1 20\n FR 65%%: %s depth num_neuron 4 2 5 3 20\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
		return 0;
	}
	const int depth=atoi(argv[1]);
	const int N=atoi(argv[2]);

	const int fr = atoi(argv[3]);
	const int delay_step = atoi(argv[4]);

	const int parts = atoi(argv[5]);

	real w1=0.0;
	real w2=0.0;
	real w3=0.0;
	real w4=0.0;
	int who=0;


	SplitType split = SynapseBalance;
	if (argc >= 7) {
		 split = (SplitType)atoi(argv[6]);
	}

	AlgoPara para = {0, 0, 0, 0};
	if (argc == 11) {
		para.syn_weight = strtof(argv[7], NULL);
		para.comm_weight = strtof(argv[8], NULL);
		para.send_weight = strtof(argv[9], NULL);
		para.recv_weight = strtof(argv[10], NULL);
	}

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

	// const real run_time=1000e-3;
	const real dt=1e-4;


	Network c(dt);

	if (node_id == 0) {

		Population *g[depth+1];

		g[1]=c.createPopulation(1, N, LIFNeuron(
						fv,v_rest,freset,
						c_m,tau_m,
						frefractory,tau_syn_e,tau_syn_i,
						fthreshold,i_offset, dt
						));

		for(int i=2;i<=depth;i++)
			g[i] = c.createPopulation(i, N, LIFNeuron(
							fv,v_rest,freset,
							c_m,tau_m,
							frefractory,tau_syn_e,tau_syn_i,
							fthreshold,0, dt
							));

		real * weight6 = getConstArray((real)(1e-9)*w1 /N, N*N);
		real * weight6_30 = getConstArray((real)(1e-9)*w2 /N, N*N);
		real * weight5 = getConstArray((real)(1e-9)*w3 /N, N*N);
		real * weight3 = getConstArray((real)(-1e-9)*w4 /N, N*N);
		real * delay = getConstArray((real)(delay_step * dt), N*N);

		enum SpikeType type=Inh;
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
			p[i] = c.createPopulation(i+depth, N, LIFNeuron(
							fv,v_rest,freset,
							c_m,tau_m,
							frefractory,tau_syn_e,tau_syn_i,
							fthreshold,0, dt
							));
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


	MNSim mn(&c, dt);	//gpu
	if (node_id == 0) {
		char name[1024];
		if (argc == 11) {
			sprintf(name, "%s_%d_%d_%d_%d_%d_%d", "standard_mpi", parts, depth, N, fr, delay_step, split); 
			mn.build_net(parts, split, name, &para);
		} else if (argc == 7) {
			sprintf(name, "%s_%d_%d_%d_%d_%d_%d", "standard_mpi", parts, depth, N, fr, delay_step, split); 
			mn.build_net(parts, split, name);
		} else {
			sprintf(name, "%s_%d_%d_%d_%d_%d", "standard_mpi", parts, depth, N, fr, delay_step); 
			mn.build_net(parts);
		}
		mn.save_net(name);
		// mn.run(run_time);	
	}

	MPI_Barrier(MPI_COMM_WORLD);

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
