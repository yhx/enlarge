/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../../include/BSim.h"

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	const int N = 3;
	Network c;
        //LIFNeuron::LIFNeuron(ID id, real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)
	Population *pn1 = c.createPopulation(0, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 10.0e-1), 1.0, 1.0));
	Population *pn2 = c.createPopulation(1, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
	Population *pn3 = c.createPopulation(2, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
	Population *pn4 = c.createPopulation(3, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
	Population *pn5 = c.createPopulation(4, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
	real * weight = getRandomArray((real)20e-3, N*N);
	real * weight2 = getRandomArray((real)15e-3, N*N);
	real * delay = getConstArray((real)1e-3, N*N);

	char v_filename[512]; 
	sprintf(v_filename, "weight1_%d.data", network->_nodeIdx);  


	saveArray("weight1.csv", weight, N*N);
	saveArray("weight2.csv", weight2, N*N);

	c.connect(pn1, pn2, weight, delay, NULL, N*N);
	c.connect(pn2, pn3, weight2, delay, NULL, N*N);
	c.connect(pn3, pn4, weight2, delay, NULL, N*N);
	c.connect(pn4, pn5, weight2, delay, NULL, N*N);

	// STSim st(&c, 1.0e-3);
	// st.run(0.1);
	
	MNSim mn(&c, 1.0e-3);
	// mn.init(&argc, &argv);
	mn.run(0.1);
	
	return 0;
} 
