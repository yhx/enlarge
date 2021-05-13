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
	Network c(DT);

	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	// char v_filename[512]; 

	if (node_id == 0) {
		//LIFNeuron::LIFNeuron(ID id, real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)
		Population *pn0 = c.createPopulation(0, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0002, 1.0, 1.0, 15.0e-3, 10.0e-1, DT));
		Population *pn1 = c.createPopulation(1, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0001, 1.0, 1.0, 15.0e-3, 0, DT));
		Population *pn2 = c.createPopulation(2, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 10.0e-1, DT));
		Population *pn3 = c.createPopulation(2, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0, DT));
		Population *pn4 = c.createPopulation(3, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0, DT));
		Population *pn5 = c.createPopulation(4, N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0, DT));

		real * weight0 = loadArray(W0_NAME, N*N);
		real * weight1 = loadArray(W1_NAME, N*N);
		real * weight2 = loadArray(W2_NAME, N*N);
		real * delay1 = loadArray(D1_NAME, N*N);
		// real * delay2 = loadArray(D2_NAME, N*N);

		SpikeType type=Inh;
		SpikeType *ii = getConstArray(type, N*N);

		c.connect(pn0, pn1, weight0, delay1, NULL, N*N);
		c.connect(pn0, pn3, weight0, delay1, NULL, N*N);
		c.connect(pn2, pn3, weight2, delay1, ii, N*N);
		c.connect(pn1, pn4, weight0, delay1, ii, N*N);
		c.connect(pn3, pn4, weight1, delay1, NULL, N*N);
		c.connect(pn3, pn5, weight1, delay1, NULL, N*N);
	}


	// STSim st(&c, 1.0e-3);
	// st.run(0.1);
	
	MNSim mn(&c, DT);
	// mn.mpi_init(&argc, &argv);
	FireInfo log;
	mn.run(SIM_TIME, true);
	
	return 0;
} 
