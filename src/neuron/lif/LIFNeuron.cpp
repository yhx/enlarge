
#include "math.h"

#include "../../third_party/json/json.h"
#include "LIFNeuron.h"
#include "LIFData.h"


LIFNeuron::LIFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, int n) : Neuron(LIF, n){
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

LIFNeuron::LIFNeuron(const LIFNeuron &neuron, int n) : Neuron(LIF, n)
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
		_Ci.insert(_Ci.end(), neuron._Ci.begin(), neuron._Ci.end());
		_Ce.insert(_Ce.end(), neuron._Ce.begin(), neuron._Ce.end());
		_C_i.insert(_C_i.end(), neuron._C_i.begin(), neuron._C_i.end());
		_C_e.insert(_C_e.end(), neuron._C_e.begin(), neuron._C_e.end());
		_v_tmp.insert(_v_tmp.end(), neuron._v_tmp.begin(), neuron._v_tmp.end());
		_V_thresh.insert(_V_thresh.end(), neuron._V_thresh.begin(), neuron._V_thresh.end());
		_V_reset.insert(_V_reset.end(), neuron._V_reset[0]);

		_i_e.insert(_i_e.end(),  neuron._i_e.begin(), neuron._i_e.end());
		_i_i.insert(_i_i.end(),  neuron._i_i.begin(), neuron._i_i.end());
	}

}

LIFNeuron::~LIFNeuron()
{
	_num = 0;
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

int LIFNeuron::packup(void * data)
{
	LIFData *p = (LIFData *) data;
	p->num = _num;
	p->pRefracTime = _refract_time.data();
	p->pRefracStep = _refract_step.data();

	p->pI_e = _i_e.data();
	p->pI_i = _i_i.data();
	p->pCe = _Ce.data();
	p->pV_reset = _V_reset.data();
	// p->pV_e = _v_.data();
	p->pV_tmp = _v_tmp.data();
	p->pI_i = _i_i.data();
	p->pV_thresh = _V_thresh.data();
	p->pCi = _Ci.data();
	p->pV_m = _v.data();
	p->pC_e = _C_e.data();
	p->pC_m = _Cm.data();
	p->pC_i = _C_i.data();
	p->is_view = true;

	return 0;
}
