/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include <string.h>

#include "../../include/BSim.h"

using namespace std;

int main(int argc, char **argv)
{
	bool load = false;

	if ((argc > 1) && (0==strcmp(argv[1], "load"))) {
		load = true;
	}

	const int N = 500;

	real dt = 1.0e-4;
	Network c(dt);
	//createPopulation(int id, int N, LIFNeuron(ID id, real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)), ID(0, 0), real tau_syn_E, real tau_syn_I);
	Population *pn0 = c.createPopulation(0, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0002, 1.0, 1.0, 15.0e-3, 10.0e-1, dt));
	Population *pn1 = c.createPopulation(1, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0001, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
	Population *pn2 = c.createPopulation(2, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 10.0e-1, dt));
	Population *pn3 = c.createPopulation(3, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
	Population *pn4 = c.createPopulation(4, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
	Population *pn5 = c.createPopulation(5, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));

	real * weight0 = NULL;
	real * weight1 = NULL;
	real * weight2 = NULL;
	real * delay1 = NULL;
	real * delay2 = NULL;

	if (load) { 
		printf("LOAD DATA...\n");
		weight0 = loadArray("weight0.csv", N*N);
		weight1 = loadArray("weight1.csv", N*N);
		delay1 = loadArray("delay1.csv", N*N);
		delay2 = loadArray("delay2.csv", N*N);
		printf("LOAD DATA FINISHED\n");
	} else {
		printf("GENERATE DATA...\n");
		//real * array = getConstArray(weight_value, num);
		weight0 = (real*)malloc_c<real>(N * N);
		weight1 = (real*)malloc_c<real>(N * N);
		weight2 = (real*)malloc_c<real>(N * N);

		// for (int i=0; i<N*N; i++) {
		//     weight0[i] = 0.0001 + (double)(i)/(double)(N*N) * 0.001;
		//     weight1[i] = 0.0011 - (double)(i)/(double)(N*N) * 0.001;
		//     weight2[i] = -0.00022 + (double)(i)/(double)(N*N) * 0.0002;
		// }

		//weight2 = getRandomArray((real)20e-3, N*N);
		delay1 = getConstArray((real)1e-4, N*N);
		delay2 = getConstArray((real)2e-4, N*N);
		for (int i=0; i<N*N; i++) {
		    weight0[i] = 0.0001 + (double)(i)/(double)(N*N) * 0.001;
		    weight1[i] = 0.0011 - (double)(i)/(double)(N*N) * 0.001;
		    weight2[i] = -0.00022 + (double)(i)/(double)(N*N) * 0.0002;
			delay1[i] = 1e-4 * (i % 30 + 1);
		}
		printf("GENERATE DATA FINISHED\n");
	}

	enum SpikeType type=Inh;
	SpikeType *ii = getConstArray(type, N*N);

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c.connect(pn0, pn1, weight0, delay1, NULL, N*N);
	c.connect(pn0, pn3, weight0, delay1, NULL, N*N);
	c.connect(pn2, pn3, weight2, delay1, ii, N*N);
	c.connect(pn1, pn4, weight0, delay1, NULL, N*N);
	c.connect(pn3, pn4, weight1, delay1, NULL, N*N);
	c.connect(pn3, pn5, weight1, delay1, NULL, N*N);

	// STSim st(&c, 1.0e-4);
	// st.run(0.1);
	SGSim sg(&c, 1.0e-4);
	sg.run(0.1);

	if (!load) {
		printf("SAVE DATA...\n");
		saveArray("weight0.csv", weight0, N*N);
		saveArray("weight1.csv", weight1, N*N);
		saveArray("delay1.csv", delay1, N*N);
		saveArray("delay2.csv", delay2, N*N);
		printf("SAVE DATA FINISHED\n");
	}
	
	return 0;
} 
