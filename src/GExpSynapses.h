/* This header file is writen by qp09
 * usually just for fun
 * Mon December 14 2015
 */
#ifndef GEXPSYNAPSES_H
#define GEXPSYNAPSES_H

#include "ID.h"

struct GExpSynapses {
	int num;
	ID *pID;
	SpikeType *pType;
	real *p_weight;
	real *p_delay;
	real *p_C1;
	real *p__C1;
	real *p_tau_syn;
	real *p_I_syn;
	real *p__dt;
	int *pSrc;
	int *pDst;

	int allocSynapses(int S);
	int allocGSynapses(GExpSynapses *pGpuSynapses);
};

int freeGSynapses(GExpSynapses *pGpuSynapses);

#endif /* GEXPSYNAPSES_H */
