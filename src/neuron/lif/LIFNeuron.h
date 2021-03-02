#ifndef LIFNEURON_H
#define LIFNEURON_H

#include <stdio.h>
#include "../../interface/Neuron.h"

class LIFNeuron : public Neuron {
public:
	LIFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, int n=0);
	LIFNeuron(const LIFNeuron &neuron, int n=0);
	~LIFNeuron();

	// virtual int recv(real I)  override;

	virtual Type getType() const override;

	// virtual int reset(SimInfo &info) override;
	// virtual int update(SimInfo &info) override;
	// virtual void monitor(SimInfo &info) override;

	// virtual size_t getSize() override;
	// virtual int getData(void *data) override;
	virtual int hardCopy(void * data, int idx, int base, const SimInfo &info) override;

	virtual Synapse * createSynapse(real weight, real delay, SpikeType type, real tau) override;

	const static Type type;
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
