
#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"

#include "PoissonSynapse.h"
#include "PoissonData.h"

// const Type PoissonSynapse::type = Poisson;

PoissonSynapse::PoissonSynapse(real weight, real delay, real tau_syn, real dt, size_t num)
	: Synapse(Poisson, num)
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

PoissonSynapse::PoissonSynapse(const real *weight, const real *delay, const real *tau_syn, real dt, size_t num)
	: Synapse(Poisson, num)
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

PoissonSynapse::PoissonSynapse(const real *weight, const real *delay, const real tau_syn, real dt, size_t num)
	: Synapse(Poisson, num)
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

PoissonSynapse::PoissonSynapse(const PoissonSynapse &s, size_t num) : Synapse(Poisson, 0)
{
	append(dynamic_cast<const Synapse *>(&s), num);
}

PoissonSynapse::~PoissonSynapse()
{
	_num = 0;
	_delay.clear();
	_weight.clear();
}

int PoissonSynapse::append(const Synapse *syn, size_t num) 
{
	const PoissonSynapse *s = dynamic_cast<const PoissonSynapse *>(syn);
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

void * PoissonSynapse::packup()
{
	PoissonData *p = static_cast<PoissonData *>(mallocPoisson());

	p->num = _num;
	p->pWeight = _weight.data();
	p->is_view = true;

	return p;
}

int PoissonSynapse::packup(void *data, size_t dst, size_t src)
{
	PoissonData *p = static_cast<PoissonData *>(data);

	p->pWeight[dst] = _weight[src];

	return 0;
}
