#include "../../include/BSim.h"
#include <stdlib.h>
#include<time.h>

using namespace std;

int main(int argc, char **argv)
{
	time_t start,end;
	start=clock(); //time(NULL);
	if(argc != 5)
	{
		printf("Need 4 paras. For example\n FR 1%%: %s depth num_neuron fire_rate delay\n", argv[0]);
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
			w1 = 2.5;
			w2 = 0.8;
			w3 = 1.5;
			w4 = 2.1;
			who = 6;
	};

	const real i_offset =6.5e-9;//*nA

	printf("depth=%d N=%d\n", depth, N);
	printf("w1=%f w2=%f w3=%f w4=%f who=%d\n", w1, w2, w3, w4, who);
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

	const real run_time=1000e-3;
	const real dt=1e-4;


	Network c(dt);


	int tree_size = (int)pow(2.0, depth+1) - 1;
	Population **g = new Population *[tree_size];

	g[0]=c.createPopulation(1, N, LIFNeuron(
				fv,v_rest,freset,
				c_m,tau_m,
				frefractory,tau_syn_e,tau_syn_i,
				fthreshold,i_offset, dt
				));

	for(int i=1;i<tree_size;i++)
		g[i] = c.createPopulation(i, N, LIFNeuron(
					fv,v_rest,freset,
					c_m,tau_m,
					frefractory,tau_syn_e,tau_syn_i,
					fthreshold,0, dt
					));

	real * weight1 = getConstArray((real)(1e-9)*w1 /N, N*N);
	real * weight2 = getConstArray((real)(1e-9)*w2 /N, N*N);
	real * weight3 = getConstArray((real)(1e-9)*w3 /N, N*N);
	real * weight4 = getConstArray((real)(-1e-9)*w4 /N, N*N);
	real * delay = getConstArray((real)(delay_step * dt), N*N);

	enum SpikeType type=Inh;
	SpikeType *ii = getConstArray(type, N*N);

	for(int i=1;i<depth;i++)
	{
		int start = (int)pow(2.0, i) - 1;
		int end = (int)pow(2.0, i+1) - 1;
		if ((i-1)%3 == 1) {
			for (int j=start; j<end; j++) {
				c.connect(g[(j-1)/2], g[j], weight2, delay, NULL, N*N);
			}
		} else if ((i-1)%3 == 2) {
			for (int j=start; j<end; j++) {
				c.connect(g[(j-1)/2], g[j], weight3, delay, NULL, N*N);
			}
		} else {
			for (int j=start; j<end; j++) {
				if (depth > 1) {
					c.connect(g[(j-1)/2], g[j], weight4, delay, ii, N*N);
				}
				c.connect(g[0], g[j], weight1, delay, NULL, N*N);
			}
		}
	}


	delArray(weight1);
	delArray(weight2);
	delArray(weight3);
	delArray(weight4);
	delArray(delay);
	delArray(ii);


	// SGSim sg(&c, dt);	//gpu
	// sg.run(run_time);	

	STSim st(&c, dt);	//gpu
	st.run(run_time);	

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
