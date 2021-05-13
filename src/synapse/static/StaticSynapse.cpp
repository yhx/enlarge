
#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"

#include "StaticSynapse.h"
#include "StaticData.h"

// const Type StaticSynapse::type = Static;

StaticSynapse::StaticSynapse(real weight, real delay, real tau_syn, real dt, size_t num)
	: Synapse(Static, num)
{
	int delay_steps = static_cast<int>(round(delay/dt));
	assert(fabs(tau_syn) > ZERO);
	if (fabs(tau_syn) > ZERO) {
		real c1 = exp(-(delay-dt*delay_steps)/tau_syn);
		weight = weight * c1;
	}

	_weight.insert(_weight.end(), num, weight);
	_delay.insert(_delay.end(), num, delay_steps);
	assert(_num == _weight.size());
}

StaticSynapse::StaticSynapse(const real *weight, const real *delay, const real *tau_syn, real dt, size_t num)
	: Synapse(Static, num)
{
	_weight.resize(num);
	_delay.resize(num);

	for (size_t i=0; i<num; i++) {
		int delay_steps = static_cast<int>(round(delay[i]/dt));
		assert(fabs(tau_syn[i]) > ZERO);
		real w = weight[i];
		if (fabs(tau_syn[i]) > ZERO) {
			real c1 = exp(-(delay[i]-dt*delay_steps)/tau_syn[i]);
			w = w * c1;
		}
		_weight[i] = w;
		_delay[i] = delay_steps;
	}

	assert(_num == _weight.size());
}

StaticSynapse::StaticSynapse(const real *weight, const real *delay, const real tau_syn, real dt, size_t num)
	: Synapse(Static, num)
{
	_weight.resize(num);
	_delay.resize(num);

	for (size_t i=0; i<num; i++) {
		int delay_steps = static_cast<int>(round(delay[i]/dt));
		real w = weight[i];
		if (fabs(tau_syn) > ZERO) {
			real c1 = exp(-(delay[i]-dt*delay_steps)/tau_syn);
			w = w * c1;
		}
		_weight[i] = w;
		_delay[i] = delay_steps;
	}

	assert(_num == _weight.size());
}

StaticSynapse::StaticSynapse(const StaticSynapse &s, size_t num) : Synapse(Static, 0)
{
	append(dynamic_cast<const Synapse *>(&s), num);
}

StaticSynapse::~StaticSynapse()
{
	_num = 0;
	_delay.clear();
	_weight.clear();
}

int StaticSynapse::append(const Synapse *syn, size_t num) 
{
	const StaticSynapse *s = dynamic_cast<const StaticSynapse *>(syn);
	size_t ret = 0;
	if ((num > 0) && (num != s->size())) {
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

void * StaticSynapse::packup()
{
	StaticData *p = static_cast<StaticData *>(mallocStatic());

	p->num = _num;
	p->pWeight = _weight.data();
	p->is_view = true;

	return p;
}

int StaticSynapse::packup(void *data, size_t dst, size_t src)
{
	StaticData *p = static_cast<StaticData *>(data);

	p->pWeight[dst] = _weight[src];

	return 0;
}
