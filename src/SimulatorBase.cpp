/* This program is writen by qp09.
 * usually just for fun.
 * Sun December 13 2015
 */

#include "SimulatorBase.h"

SimulatorBase::SimulatorBase(Network* network, real dt)
{
	this->network = network;
	this->dt = dt;
}

SimulatorBase::~SimulatorBase()
{
}

int SimulatorBase::reset()
{
	SimInfo info = { 0, dt };
	vector<SynapseBase*>::iterator iterS;
	vector<NeuronBase*>::iterator iterN;
	vector<PopulationBase*>::iterator iterP;
	for (iterS=network->pSynapses.begin(); iterS!=network->pSynapses.end(); iterS++) {
		SynapseBase *p = *iterS;
		p->reset(info);
	}
	for (iterN=network->pNeurons.begin(); iterN!=network->pNeurons.end(); iterN++) {
		NeuronBase * p = *iterN;
		p->reset(info);
	}
	for (iterP=network->pPopulations.begin(); iterP!=network->pPopulations.end(); iterP++) {
		PopulationBase * p = *iterP;
		p->reset(info);
	}
	return 0;
}