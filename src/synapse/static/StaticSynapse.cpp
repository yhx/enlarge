
#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"

#include "StaticSynapse.h"
#include "StaticData.h"

// const Type StaticSynapse::type = Static;

StaticSynapse::StaticSynapse(real weight, real delay, real tau_syn, real dt, int num)
	: Synapse(Static, num)
{
	int delay_steps = static_cast<int>(round(_delay/dt));
	assert(fabs(tau_syn) > ZERO);
	if (fabs(tau_syn) > ZERO) {
		real c1 = exp(-(delay-dt*delay_steps)/tau_syn);
		weight = weight * c1;
	}

	_weight.insert(_weight.end(), num, weight);
	_delay.insert(_delay.end(), num, delay_steps);
	assert(_num == _weight.size());
}

StaticSynapse::StaticSynapse(const StaticSynapse &s, int num) : Synapse(Static, 0)
{
	append(dynamic_cast<const Synapse *>(&s), num);
}

StaticSynapse::~StaticSynapse()
{
	_num = 0;
	_delay.clear();
	_weight.clear();
}

int StaticSynapse::append(const Synapse *syn, int num) 
{
	const StaticSynapse *s = dynamic_cast<const StaticSynapse *>(syn);
	int ret = 0;
	if (num > 0) {
		ret = num;
		_weight.insert(_weight.end(), num, s->_weight[0]);
		_delay.insert(_delay.end(), num, s->_delay[0]);
	} else {
		ret = s->_num;
		_weight.insert(_weight.end(), s->_weight.begin(), s->_weight.end());
		_delay.insert(_delay.end(), s->_delay.begin(), s->_delay.end());
	}
	_num += ret;
	assert(_num == _weight.size());

	return ret;
}

int StaticSynapse::packup(void * data)
{
	StaticData *p = (StaticData *) data;

	p->pWeight = _weight.data();
	// p->p_src[idx] = this->getSrc()->getID();
	// p->pDst[idx] = this->getDst()->getID();

	return 1;
}

