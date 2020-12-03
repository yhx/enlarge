
#include "Neuron.h"

Neuron::~Neuron() 
{
	pSynapses.clear();
}

Synapse * Neuron::addSynapse(Synapse * synapse)
{
	pSynapses.push_back(synapse);
	return synapse;
}

vector<Synapse*> & Neuron::getSynapses() 
{
	return pSynapses;
}

