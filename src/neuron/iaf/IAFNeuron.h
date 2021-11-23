#ifndef IAFNEURON_H
#define IAFNEURON_H

#include <stdio.h>
#include "../../interface/Neuron.h"

class IAFNeuron : public Neuron {
public:
	/**
	 * constructor of IAFNeuron.
	 * dt: simulation time per circle
	 * tau_refrac: fractory time
	 * num: neuron number in the population
	 * other params see `IAFData.h`.
	 **/
	// IAFNeuron(real dt, size_t num=1, real Tau=10.0, real C=250.0, real t_ref=2.0*1e-3, real E_L=-70.0
	// 		, real I_e=0.0, real Theta=15.0, real V_reset=0.0, real tau_ex=2.0, real tau_in=2.0, real rho=0.01,
	// 		real delta=0.0, real V_m=0, real i_0=0.0, real i_1=0.0, real i_syn_in=0.0, real i_syn_ex=0.0);
	IAFNeuron(real dt, size_t num=1, real Tau=10.0, real C=250.0, real t_ref=2.0*1e-3, real E_L=-65.0
			, real I_e=0.0, real Theta=-50.0, real V_reset=-65.0, real tau_ex=0.5, real tau_in=0.5, real rho=0.01,
			real delta=0.0, real V_m=-58.0, real i_0=0.0, real i_1=0.0, real i_syn_in=0.0, real i_syn_ex=0.0);
	// IAFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, size_t num=1);
	IAFNeuron(const IAFNeuron &n, size_t num=0);
	~IAFNeuron();

	virtual int append(const Neuron *n, size_t num=0) override;
	virtual void * packup() override;
	int packup(void *data, size_t dst, size_t src) override;
	real propagator_32(real tau_syn, real tau, real C, real dt);

protected:
	vector<int> _refract_step;
	vector<int> _refract_time;

	// neuron model param
	vector<real> _Tau;
	vector<real> _C;
	// vector<real> _t_ref;
	vector<real> _E_L;
	vector<real> _I_e;
	vector<real> _Theta;
	vector<real> _V_reset;
	vector<real> _tau_ex;
	vector<real> _tau_in;
	vector<real> _rho;
	vector<real> _delta;

	// state param
	vector<real> _i_0;
	vector<real> _i_1;
	vector<real> _i_syn_ex;
	vector<real> _i_syn_in;
	vector<real> _V_m;

	// internal param
	vector<real> _P11ex;
	vector<real> _P11in;
	vector<real> _P22;
	vector<real> _P21ex;
	vector<real> _P21in;
	vector<real> _P20;
};

#endif /* IAFNEURON_H */
