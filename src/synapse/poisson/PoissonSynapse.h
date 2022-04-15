#ifndef POISSONSYNAPSE_H
#define POISSONSYNAPSE_H

#include <stdio.h>
#include <list>
#include "../../interface/Synapse.h"

using std::list;

class PoissonSynapse : public Synapse {
public:
	// 构造函数和析构函数
	PoissonSynapse(real mean, real weight, real delay, real tau_syn, real dt, size_t num=1);
	PoissonSynapse(const real *mean, const real *weight, const real *delay, const real *tau_syn, real dt, size_t num=1);
	PoissonSynapse(const real *mean, const real *weight, const real *delay, const real tau_syn, real dt, size_t num=1);
	PoissonSynapse(const PoissonSynapse &s, size_t num=0);
	~PoissonSynapse();

	virtual int append(const Synapse *s, size_t num=0) override;

	virtual void * packup() override;
	virtual int packup(void *data, size_t dst, size_t src) override;

	virtual real weight(size_t idx) override {
		return _weight[idx];
	}

protected:
	vector<real> _weight;
	vector<real> _mean; 	// expectation of poisson distribution
	// const static Type type;
	// real _weight;
	// real _delay;
	// real _tau_syn;
	// list<int> delay_queue;
	// NeuronBase *pDest;
};

#endif /* STATICSYNAPSE_H */
