#ifndef STATICSYNAPSE_H
#define STATICSYNAPSE_H

#include <stdio.h>
#include <list>
#include "../../interface/Synapse.h"

using std::list;

class StaticSynapse : public Synapse {
public:
	StaticSynapse(real weight, real delay, real tau_syn, real dt, size_t num=1);
	StaticSynapse(const real *weight, const real *delay, const real *tau_syn, real dt, size_t num=1);
	StaticSynapse(const StaticSynapse &s, size_t num=0);
	~StaticSynapse();

	virtual int append(const Synapse *s, size_t num=0) override;

	virtual int packup(void * data) override;

protected:
	vector<real> _weight;
	// const static Type type;
	// real _weight;
	// real _delay;
	// real _tau_syn;
	// list<int> delay_queue;
	// NeuronBase *pDest;
};

#endif /* STATICSYNAPSE_H */
