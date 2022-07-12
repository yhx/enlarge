#ifndef GSTATICSYNAPSES_H
#define GSTATICSYNAPSES_H

#include "../../utils/type.h"
#include "../../utils/macros.h"

struct GStaticSynapses {
	real *p_weight;
	//int *p_delay;
	//int *p_src;
	int *p_dst;
};

SYNAPSE_GPU_FUNC_DEFINE(Static)

#endif /* GSTATICSYNAPSES_H */
