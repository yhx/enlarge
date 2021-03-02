
#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"

#include "StaticSynapse.h"
#include "StaticData.h"

const Type StaticSynapse::type = Static;

StaticSynapse::StaticSynapse(real weight, real delay, real tau_syn)
	: Synapse(0, weight, delay), _tau_syn(tau_syn)
{
}

StaticSynapse::StaticSynapse(const StaticSynapse &synapse) : Synapse()
{
	this->_weight = synapse._weight;
	this->_delay = synapse._delay;
	this->_tau_syn = synapse._tau_syn;
}

StaticSynapse::~StaticSynapse()
{
	// delay_queue.clear();
}

// int StaticSynapse::recv()
// {
// 	delay_queue.push_back(_delay_steps);
// 
// 	return 0;
// }

Type StaticSynapse::getType() const
{
	return type;
}


int StaticSynapse::packup(void * data, int idx, int base, const SimInfo &info)
{
	StaticData *p = (StaticData *) data;

	real dt = info.dt;
	int delay_steps = static_cast<int>(round(_delay/dt));
	real weight = this->_weight;
	assert(fabs(_tau_syn) > ZERO);
	if (fabs(_tau_syn) > ZERO) {
		real c1 = exp(-(_delay-dt*delay_steps)/_tau_syn);
		weight = weight * c1;
	}
	setID(idx+base);

	p->pWeight[idx] = weight;
	//p->p_src[idx] = this->getSrc()->getID();
	p->pDst[idx] = this->getDst()->getID();

	return 1;
}

