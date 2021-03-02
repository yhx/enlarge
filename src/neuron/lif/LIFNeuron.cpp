
#include "math.h"

#include "../../third_party/json/json.h"
#include "LIFNeuron.h"
#include "LIFData.h"

const Type LIFNeuron::type = LIF;

LIFNeuron::LIFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, int n) : Neuron(){
	_num = n;

	real rm = (fabs(cm) > ZERO)?(tau_m/cm):1.0;
	real Cm = (tau_m>0)?exp(-dt/tau_m):0.0;
	real Ce = (tau_syn_E > 0)?exp(-dt/tau_syn_E):0.0;
	real Ci = (tau_syn_I > 0)?exp(-dt/tau_syn_I):0.0;

	real v_tmp = i_offset * rm + v_rest;
	v_tmp *= (1-Cm);

	real C_e = rm * tau_syn_E/(tau_syn_E - tau_m);
	real C_i = rm * tau_syn_I/(tau_syn_I - tau_m);

	C_e = C_e * (Ce - Cm);
	C_i = C_i * (Ci - Cm);
	
	int refract_time = static_cast<int>(tau_refrac/dt);

	_refract_step.insert(_refract_step.end(), n, 0);
	_refract_time.insert(_refract_time.end(), n, refract_time);

	_v.insert(_v.end(), n, v_init);
	_Ci.insert(_Ci.end(), n, Ci);
	_Ce.insert(_Ce.end(), n, Ce);
	_C_i.insert(_C_i.end(), n, C_i);
	_C_e.insert(_C_e.end(), n, C_e);
	_v_tmp.insert(_v_tmp.end(), n, v_tmp);
	_V_thresh.insert(_V_thresh.end(), n, v_thresh);
	_V_reset.insert(_V_reset.end(), n, v_reset);

	_i_e.insert(_i_e.end(), n, 0);
	_i_i.insert(_i_i.end(), n, 0);
}

LIFNeuron::LIFNeuron(const LIFNeuron &neuron, int n) : Neuron()
{
	if (n > 0) {
		_num = n;
		_refract_step.insert(_refract_step.end(), n, 0);
		_refract_time.insert(_refract_time.end(), n, neuron._refract_time[0]);

		_v.insert(_v.end(), n, neuron._v[0]);
		_Ci.insert(_Ci.end(), n, neuron._Ci[0]);
		_Ce.insert(_Ce.end(), n, neuron._Ce[0]);
		_C_i.insert(_C_i.end(), n, neuron._C_i[0]);
		_C_e.insert(_C_e.end(), n, neuron._C_e[0]);
		_v_tmp.insert(_v_tmp.end(), n, neuron._v_tmp[0]);
		_V_thresh.insert(_V_thresh.end(), n, neuron._V_thresh[0]);
		_V_reset.insert(_V_reset.end(), n, neuron._V_reset[0]);

		_i_e.insert(_i_e.end(), n, 0);
		_i_i.insert(_i_i.end(), n, 0);
	} else {
		_num = neuron._num;
		_refract_step.insert(_refract_step.end(), neuron._v.begin(), neuron._v.end());
		_refract_time.insert(_refract_time.end(), neuron._refract_time.begin(), neuron._refract_time.end());

		_v.insert(_v.end(), neuron._v.begin(), neuron._v.begin());
		_Ci.insert(_Ci.end(), neuron._Ci[0].begin(), neuron._Ci[0].end());
		_Ce.insert(_Ce.end(), neuron._Ce[0].begin(), neuron._Ce[0].end());
		_C_i.insert(_C_i.end(), neuron._C_i.begin(), neuron._C_i.end());
		_C_e.insert(_C_e.end(), neuron._C_e.begin(), neuron._C_e.end());
		_v_tmp.insert(_v_tmp.end(), neuron._v_tmp[0]);
		_V_thresh.insert(_V_thresh.end(), neuron._V_thresh[0]);
		_V_reset.insert(_V_reset.end(), neuron._V_reset[0]);

		_i_e.insert(_i_e.end(),  0);
		_i_i.insert(_i_i.end(),  0);
	}

}

LIFNeuron::~LIFNeuron()
{
	num = 0;
	_refract_step.clear();
	_refract_time.clear();

	_v.clear();
	_Ci.clear();
	_Ce.clear();
	_Cm.clear();
	_C_i.clear();
	_C_e.clear();
	_v_tmp.clear();
	_V_thresh.clear();
	_V_reset.clear();

	_i_i.clear();
	_i_e.clear();
}

Type LIFNeuron::getType() const
{
	return type;
}

// int LIFNeuron::reset(SimInfo &info)
// {
	// real dt = info.dt;

	// real rm = 1.0;
	// if (fabs(_cm) > ZERO) {
	// 	rm = _tau_m/_cm;
	// }
	// if (_tau_m > 0) {
	// 	_Cm = exp(-dt/_tau_m);
	// } else {
	// 	_Cm = 0.0;
	// }

	// if (_tau_syn_E > 0) {
	// 	_CE = exp(-dt/_tau_syn_E);
	// } else {
	// 	_CE = 0.0;
	// }

	// if (_tau_syn_I > 0) {
	// 	_CI = exp(-dt/_tau_syn_I);
	// } else {
	// 	_CI = 0.0;
	// }

	// _v_tmp = _i_offset * rm + _v_rest;
	// _v_tmp *= (1-_Cm);

	// _C_E = rm * _tau_syn_E/( _tau_syn_E - _tau_m);
	// _C_I = rm * _tau_syn_I/( _tau_syn_I - _tau_m);

	// _C_E = _C_E * (_CE - _Cm);
	// _C_I = _C_I * (_CI - _Cm);

	// _refrac_time = static_cast<int>(_tau_refrac/dt);
	// _refrac_step = 0;

	// _i_I = 0;
	// _i_E = 0;

	// _vm = _v_init;

	// return 0;
// }

// void LIFNeuron::monitor(SimInfo &info)
// {
// }

// size_t LIFNeuron::getSize()
// {
// 	return sizeof(GLIFNeurons);
// }

Synapse * LIFNeuron::createSynapse(real weight, real delay, SpikeType type, real tau) {
	printf("Not implemented!\n");
	return NULL;
}

int LIFNeuron::hardCopy(void * data, int idx, int base, const SimInfo &info)
{
	LIFData *p = (LIFData *) data;

	real dt = info.dt;
	real rm = (fabs(_cm) > ZERO)?(_tau_m/_cm):1.0;
	real Cm = (_tau_m>0)?exp(-dt/_tau_m):0.0;
	real Ce = (_tau_syn_E > 0)?exp(-dt/_tau_syn_E):0.0;
	real Ci = (_tau_syn_I > 0)?exp(-dt/_tau_syn_I):0.0;

	real v_tmp = _i_offset * rm + _v_rest;
	v_tmp *= (1-Cm);

	real C_e = rm * _tau_syn_E/( _tau_syn_E - _tau_m);
	real C_i = rm * _tau_syn_I/( _tau_syn_I - _tau_m);

	C_e = C_e * (Ce - Cm);
	C_i = C_i * (Ci - Cm);

	setID(idx+base);
	p->pV_m[idx] = _v_init;
	p->pCi[idx] = Ci;
	p->pCe[idx] = Ce;
	p->pC_i[idx] = C_i;
	p->pC_e[idx] = C_e;
	p->pV_tmp[idx] = v_tmp;
	p->pI_i[idx] = 0;
	p->pI_e[idx] = 0;
	p->pV_thresh[idx] = _v_thresh;
	p->pV_reset[idx] = _v_reset;
	p->pC_m[idx] = Cm;

	p->pRefracStep[idx] = 0;
	p->pRefracTime[idx] = static_cast<int>(_tau_refrac/dt);;
	//p->p_start_I[idx] = this->_start_I;
	//p->p_start_E[idx] = this->_start_E;
	//p->p_end[idx] = this->_end;
	return 1;
}

// int LIFNeuron::getData(void *data)
// {
// 	Json::Value *p = (Json::Value *)data;
// 	(*p)["id"] = getID();
// 	(*p)["v_init"] = _v_init;
// 	(*p)["v_rest"] = _v_rest;
// 	(*p)["_v_reset"] = _v_reset;
// 	(*p)["_cm"] = _cm;
// 	(*p)["_tau_n"] = _tau_m;
// 	(*p)["_tau_refrac"] = _tau_refrac;
// 	(*p)["_tau_syn_E"] = _tau_syn_E;
// 	(*p)["_tau_syn_I"] = _tau_syn_I;
// 	(*p)["v_thresh"] = _v_thresh;
// 	(*p)["i_offset"] = _i_offset;
// 
// 	return 0;
// }
