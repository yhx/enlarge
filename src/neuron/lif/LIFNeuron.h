#ifndef LIFNEURON_H
#define LIFNEURON_H

#include <stdio.h>
#include "../../interface/Neuron.h"

class LIFNeuron : public Neuron {
public:
	LIFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, size_t num=1);
	LIFNeuron(const LIFNeuron &n, size_t num=0);
	~LIFNeuron();

	virtual int append(const Neuron *n, size_t num=0) override;
	virtual void * packup() override;
	int packup(void *data, size_t dst, size_t src) override;

protected:
	vector<int> _refract_step;
	vector<int> _refract_time;

	vector<real> _v;
	vector<real> _Ci;
	vector<real> _Ce;
	vector<real> _Cm;
	vector<real> _C_i;
	vector<real> _C_e;
	vector<real> _v_tmp;
	vector<real> _V_thresh;
	vector<real> _V_reset;

	vector<real> _i_i;
	vector<real> _i_e;
	
};

#endif /* LIFNEURON_H */
