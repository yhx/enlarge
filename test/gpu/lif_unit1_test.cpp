/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../../include/BSim.h"

using namespace std;

int main(int argc, char **argv)
{
	const int N = 10;
	real dt = 1.0e-4;
	Network c(1.0e-4);
	//createPopulation(int id, int N, LIFNeuron(ID id, real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)), ID(0, 0), real tau_syn_E, real tau_syn_I);
	Population *pn0 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 100.0e-1, dt));
	Population *pn1 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
	Population *pn2 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
	//Population *pn3 = c.createPopulation(N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));

	real * weight0 = NULL;
	real * weight1 = NULL;
	real * delay = NULL;

	printf("GENERATE DATA...\n");
	weight0 = getConstArray((real)1e-5, N*N);
	weight1 = getConstArray((real)2e-5, N*N);
	delay = getConstArray((real)1e-4, N*N);
	printf("GENERATE DATA FINISHED\n");

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c.connect(pn0, pn1, weight0, delay, NULL, N*N);
	c.connect(pn1, pn2, weight1, delay, NULL, N*N);

	STSim st(&c, 1.0e-4);
	st.run(0.1);
	SGSim sg(&c, 1.0e-4);
	sg.run(0.1);

	return 0;
} 
