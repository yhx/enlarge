
#include "../../include/BSim.h"

using namespace std;

int main(int argc, char **argv)
{

	Network c;
	Population<Constant_spikes> *pn0 = c.createPopulation(0, 36, Constant_spikes(ConstantNeuron(0.5), 1.0, 1.0));
	for (int i = 0; i < pn0->getNum(); i++) {
		Constant_spikes * n = static_cast<Constant_spikes*>(pn0->getNeuron(i));
		n->setRate(i/36.0 * 0.5 + 0.5);
	}

	Population<LIF_curr_exp> *pn1 = c.createPopulation(1, 36, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));
	Population<Max_pooling> *pn2 = c.createPopulation(2, 9, Max_pooling(MaxNeuron(4), 0, 0));
	Population<LIF_curr_exp> *pn3 = c.createPopulation(3, 9, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));

	real * weight0 = NULL;
	real * weight1 = NULL;
	real * delay = NULL;

	//real * array = getConstArray(weight_value, num);
	weight0 = getConstArray((real)20e-4, 9);
	weight1 = getConstArray((real)10e-4, 81);
	//weight2 = getRandomArray((real)20e-3, N*N);
	delay = getConstArray((real)1e-3, 81);

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c.connectConv(pn0, pn1, weight0, delay, NULL, 6, 6, 3, 3);
	c.connectPooling(pn1, pn2, 1.0e-3, 6, 6, 2, 2);
	c.connect(pn2, pn3, weight1, delay, NULL, 81);
	//c.connect(pn0, pn1, weight0, delay, NULL, 6);

	STSim st(&c, 1.0e-3);
	st.run(1);

	SGSim sg(&c, 1.0e-3);
	sg.run(1);

	return 0;
} 
