
#include <assert.h>
#include <limits>

#include "math.h"

#include "../../third_party/json/json.h"
#include "../../../msg_utils/helper/helper_c.h"
#include "IAFNeuron.h"
#include "IAFData.h"
#include "../../utils/macros.h"

inline double
expm1( double x )
{
#if HAVE_EXPM1
	return ::expm1( x ); // use library implementation if available
#else
	// compute using Taylor series, see GSL
	// e^x-1 = x + x^2/2! + x^3/3! + ...
	if ( x == 0 ) {
		return 0;
	}
	if ( std::abs( x ) > std::log( 2.0 ) ) {
		return std::exp( x ) - 1;
	}
	else {
		double sum = x;
		double term = x * x / 2;
		long n = 2;
		while ( std::abs( term ) > std::abs( sum ) * std::numeric_limits< double >::epsilon() ) {
		sum += term;
		++n;
		term *= x / n;
		}
		return sum;
	}
#endif
}

real IAFNeuron::propagator_32(real tau_syn, real tau, real C, real dt) {
	const real P32_linear = 1 / (2. * C * tau * tau) * dt * dt * (tau_syn - tau) * exp( -dt / tau );
	const real P32_singular = dt / C * exp( -dt / tau );
	const real P32 = -tau / (C * (1 - tau / tau_syn)) * exp( -dt / tau_syn ) * expm1( dt * (1 / tau_syn - 1 / tau) );
	const real dev_P32 = abs(P32 - P32_singular);
	if (tau == tau_syn or (abs(tau - tau_syn) < 0.1 and dev_P32 > 2 * abs(P32_linear))) {
		return P32_singular;
	}
	else {
		return P32;
	}
}

IAFNeuron::IAFNeuron(real dt, size_t num, real Tau, real C, real t_ref, real E_L, real I_e,
			real Theta, real V_reset, real tau_ex, real tau_in, real rho,
			real delta, real V_m, real i_0, real i_1, real i_syn_in, real i_syn_ex)
			 : Neuron(IAF, num) {
	// deal with refractory
	int refract_time = static_cast<int>(t_ref/dt);
	
	_refract_step.insert(_refract_step.end(), num, 0);
	_refract_time.insert(_refract_time.end(), num, refract_time);
	
	// model param
	_Tau.insert(_Tau.end(), num, Tau);
	_C.insert(_C.end(), num, C);
	// _t_ref.insert(_t_ref.end(), num, t_ref);
	_E_L.insert(_E_L.end(), num, E_L);
	_I_e.insert(_I_e.end(), num, I_e);
	_Theta.insert(_Theta.end(), num, Theta);
	_V_reset.insert(_V_reset.end(), num, V_reset);
	_tau_ex.insert(_tau_ex.end(), num, tau_ex);
	_tau_in.insert(_tau_in.end(), num, tau_in);
	_rho.insert(_rho.end(), num, rho);
	_delta.insert(_delta.end(), num, delta);

	// state param
	_V_m.insert(_V_m.end(), num, V_m);
	_i_0.insert(_i_0.end(), num, i_0);
	_i_1.insert(_i_1.end(), num, i_1);
	_i_syn_in.insert(_i_syn_in.end(), num, i_syn_in);
	_i_syn_ex.insert(_i_syn_ex.end(), num, i_syn_ex);
	
	// calculate intermidiate value
	_P11ex.insert(_P11ex.end(), num, exp( -dt * 1e3 / tau_ex));
	_P11in.insert(_P11in.end(), num, exp( -dt * 1e3 / tau_in));
	_P22.insert(_P22.end(), num, exp( -dt * 1e3 / Tau ));
	_P21ex.insert(_P21ex.end(), num, propagator_32(tau_ex, Tau, C, dt * 1e3));
	_P21in.insert(_P21in.end(), num, propagator_32(tau_in, Tau, C, dt * 1e3));
	_P20.insert(_P20.end(), num, Tau / C * ( 1.0 - exp( -dt * 1e3 / Tau ) ));
	
	assert(t_ref > 0);
	assert(_num == _V_m.size());
}

IAFNeuron::IAFNeuron(const IAFNeuron &n, size_t num) : Neuron(IAF, 0)
{
	append(dynamic_cast<const Neuron *>(&n), num);
}

IAFNeuron::~IAFNeuron()
{
	_num = 0;
	_refract_step.clear();
	_refract_time.clear();

	// neuron model param
	 _Tau.clear();
	 _C.clear();
	//  _t_ref.clear();
	 _E_L.clear();
	 _I_e.clear();
	 _Theta.clear();
	 _V_reset.clear();
	 _tau_ex.clear();
	 _tau_in.clear();
	 _rho.clear();
	 _delta.clear();

	// state param
	 _i_0.clear();
	 _i_1.clear();
	 _i_syn_ex.clear();
	 _i_syn_in.clear();
	 _V_m.clear();
}

int IAFNeuron::append(const Neuron * neuron, size_t num)
{
	const IAFNeuron *n = dynamic_cast<const IAFNeuron *>(neuron);
	int ret = 0;
	if ((num > 0) && (num != n->size())) {
		ret = num;
		_refract_step.insert(_refract_step.end(), num, 0);
		_refract_time.insert(_refract_time.end(), num, n->_refract_time[0]);

		// model neuron
		INIT_PARAMETER(_Tau, num, n);
		// _Tau.insert(_Tau.end(), num, n->_Tau[0]);
		_C.insert(_C.end(), num, n->_C[0]);
		_E_L.insert(_E_L.end(), num, n->_E_L[0]);
		_I_e.insert(_I_e.end(), num, n->_I_e[0]);
		_Theta.insert(_Theta.end(), num, n->_Theta[0]);
		_V_reset.insert(_V_reset.end(), num, n->_V_reset[0]);
		_tau_ex.insert(_tau_ex.end(), num, n->_tau_ex[0]);
		_tau_in.insert(_tau_in.end(), num, n->_tau_in[0]);
		_rho.insert(_rho.end(), num, n->_rho[0]);
		_delta.insert(_delta.end(), num, n->_delta[0]);

		// state param
		_i_0.insert(_i_0.end(), num, n->_i_0[0]);
		_i_1.insert(_i_1.end(), num, n->_i_1[0]);
		_i_syn_ex.insert(_i_syn_ex.end(), num, n->_i_syn_ex[0]);
		_i_syn_in.insert(_i_syn_in.end(), num, n->_i_syn_in[0]);
		_V_m.insert(_V_m.end(), num, n->_V_m[0]);

		// intermidiate value
		_P11ex.insert(_P11ex.end(), num, n->_P11ex[0]);
		_P11in.insert(_P11in.end(), num, n->_P11in[0]);
		_P22.insert(_P22.end(), num, n->_P22[0]);
		_P21ex.insert(_P21ex.end(), num, n->_P21ex[0]);
		_P21in.insert(_P21in.end(), num, n->_P21in[0]);
		_P20.insert(_P20.end(), num, n->_P20[0]);
	} else {
		ret = n->size();
		_refract_step.insert(_refract_step.end(), n->_refract_step.begin(), n->_refract_step.end());
		_refract_time.insert(_refract_time.end(), n->_refract_time.begin(), n->_refract_time.end());

		// model neuron
		_Tau.insert(_Tau.end(), n->_Tau.begin(), n->_Tau.end());
		_C.insert(_C.end(), n->_C.begin(), n->_C.end());
		_E_L.insert(_E_L.end(), n->_E_L.begin(), n->_E_L.end());
		_I_e.insert(_I_e.end(), n->_I_e.begin(), n->_I_e.end());
		_Theta.insert(_Theta.end(), n->_Theta.begin(), n->_Theta.end());
		_V_reset.insert(_V_reset.end(), n->_V_reset.begin(), n->_V_reset.end());
		_tau_ex.insert(_tau_ex.end(), n->_tau_ex.begin(), n->_tau_ex.end());
		_tau_in.insert(_tau_in.end(), n->_tau_in.begin(), n->_tau_in.end());
		_rho.insert(_rho.end(), n->_rho.begin(), n->_rho.end());
		_delta.insert(_delta.end(), n->_delta.begin(), n->_delta.end());

		// state param
		_i_0.insert(_i_0.end(), n->_i_0.begin(), n->_i_0.end());
		_i_1.insert(_i_1.end(), n->_i_1.begin(), n->_i_1.end());
		_i_syn_ex.insert(_i_syn_ex.end(), n->_i_syn_ex.begin(), n->_i_syn_ex.end());
		_i_syn_in.insert(_i_syn_in.end(), n->_i_syn_in.begin(), n->_i_syn_in.end());
		_V_m.insert(_V_m.end(), n->_V_m.begin(), n->_V_m.end());

		// intermidiate value
		_P11ex.insert(_P11ex.end(), n->_P11ex.begin(), n->_P11ex.end());
		_P11in.insert(_P11in.end(), n->_P11in.begin(), n->_P11in.end());
		_P22.insert(_P22.end(), n->_P22.begin(), n->_P22.end());
		_P21ex.insert(_P21ex.end(), n->_P21ex.begin(), n->_P21ex.end());
		_P21in.insert(_P21in.end(), n->_P21in.begin(), n->_P21in.end());
		_P20.insert(_P20.end(), n->_P20.begin(), n->_P20.end());
	}

	_num += ret;
	assert(_num == _V_m.size());

	return ret;
}

void * IAFNeuron::packup()
{
	IAFData *p = static_cast<IAFData*>(mallocIAF());
	assert(p != NULL);
	p->num = _num;
	p->pRefracTime = _refract_time.data();
	p->pRefracStep = _refract_step.data();

	// neuron model param
	// p->pTau = _Tau.data();
	// p->pC = _C.data();
	// p->pt_ref = _t_ref.data();
	p->pE_L = _E_L.data();
	p->pI_e = _I_e.data();
	p->pTheta = _Theta.data();
	p->pV_reset = _V_reset.data();
	// p->ptau_ex = _tau_ex.data();
	// p->ptau_in = _tau_in.data();
	// p->prho = _rho.data();
	// p->pdelta = _delta.data();

	// state param
	p->pi_0 = _i_0.data();
	p->pi_1 = _i_1.data();
	p->pi_syn_ex = _i_syn_ex.data();
	p->pi_syn_in =  _i_syn_in.data();
	p->pV_m =  _V_m.data();

	// internal param
	p->pP11ex = _P11ex.data();
	p->pP11in = _P11in.data();
	p->pP22 = _P22.data();
	p->pP21ex = _P21ex.data();
	p->pP21in = _P21in.data();
	p->pP20 = _P20.data();

	p->_fire_count = malloc_c<int>(_num);
	p->is_view = true;

	return p;
}

int IAFNeuron::packup(void *data, size_t dst, size_t src)
{
	IAFData *p = static_cast<IAFData*>(data);
	p->pRefracTime[dst] = _refract_time[src];
	p->pRefracStep[dst] = _refract_step[src];

	// neuron model param
	// p->pTau[dst] = _Tau[src];
	// p->pC[dst] = _C[src];
	// p->pt_ref[dst] = _t_ref[src];
	p->pE_L[dst] = _E_L[src];
	p->pI_e[dst] = _I_e[src];
	p->pTheta[dst] = _Theta[src];
	p->pV_reset[dst] = _V_reset[src];
	// p->ptau_ex[dst] = _tau_ex[src];
	// p->ptau_in[dst] = _tau_in[src];
	// p->prho[dst] = _rho[src];
	// p->pdelta[dst] = _delta[src];

	// state param
	p->pi_0[dst] = _i_0[src];
	p->pi_1[dst] = _i_1[src];
	p->pi_syn_ex[dst] = _i_syn_ex[src];
	p->pi_syn_in[dst] =  _i_syn_in[src];
	p->pV_m[dst] =  _V_m[src];

	// internal param
	p->pP11ex[dst] = _P11ex[src];
	p->pP11in[dst] = _P11in[src];
	p->pP22[dst] = _P22[src];
	p->pP21ex[dst] = _P21ex[src];
	p->pP21in[dst] = _P21in[src];
	p->pP20[dst] = _P20[src];

	return 0;
}
