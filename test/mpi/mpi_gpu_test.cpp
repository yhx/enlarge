/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../../include/BSim.h"

#include "info.h"

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	Network c;

	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	// char v_filename[512]; 

	if (node_id == 0) {
		//LIFNeuron::LIFNeuron(ID id, real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)
		Population *pn1 = c.createPopulation(0, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 10.0e-1), 1.0, 1.0));
		Population *pn2 = c.createPopulation(1, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
		Population *pn3 = c.createPopulation(2, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
		Population *pn4 = c.createPopulation(3, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));
		Population *pn5 = c.createPopulation(4, N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-4, 0.0, 1.0, 1.0, 15.0e-3, 0), 1.0, 1.0));

		real * weight = loadArray(W1_NAME, N*N);
		real * weight2 = loadArray(W2_NAME, N*N);
		real * weight3 = loadArray(W3_NAME, N*N);
		real * weight4 = loadArray(W4_NAME, N*N);
		real * delay = loadArray(D_NAME, N*N);


		c.connect(pn1, pn2, weight, delay, NULL, N*N);
		c.connect(pn2, pn3, weight2, delay, NULL, N*N);
		c.connect(pn3, pn4, weight3, delay, NULL, N*N);
		c.connect(pn4, pn5, weight4, delay, NULL, N*N);
	}


	// STSim st(&c, 1.0e-3);
	// st.run(0.1);
	
	MNSim mn(&c, DT);
	// mn.mpi_init(&argc, &argv);
	FireInfo log;
	mn.run(SIM_TIME, true);
	
	return 0;
} 
