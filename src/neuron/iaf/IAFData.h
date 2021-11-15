
#ifndef IAFDATA_H
#define IAFDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct IAFData {
	bool is_view;
	size_t num;

	int *pRefracTime;
	int *pRefracStep;

	/********* new param ********/
	// model param
	/**
	 * model param:
	 * some parameters do not use in LIFData.compute.
	 * can save communication a lot of time.
	 **/
	// real *pTau;			// Membrane time constant in ms
	// real *pC;			// Membrane capacitance in pF
	// real *pt_ref;		// Refractory period in ms
	real *pE_L;			// Resting potential in mV
	real *pI_e;			// External current in pA
	real *pTheta;		// Threshold, RELATIVE TO RESTING POTENTAIL(!). I.e. the real threshold is (E_L_+Theta_).
	real *pV_reset;		// reset value of the membrane potential
	// real *ptau_ex;		// Time constant of excitatory synaptic current in ms
	// real *ptau_in;		// Time constant of inhibitory synaptic current in ms
	// real *prho;			// Stochastic firing intensity at threshold in 1/s
	// real *pdelta;		// Width of threshold region in mV

	/**
	 * state params:
	 * pi_0 & pi_1:
	 * Current input through receptor_type 0 are handled as stepwise constant
	 * current input as in other iaf models, i.e., this current directly enters
	 *  the membrane potential equation. Current input through receptor_type 1,
	 *  in contrast, is filtered through an exponential kernel with the time 
	 * constant of the excitatory synapse, tau_syn_ex. 
	 **/
    real *pi_0;		  	//! synaptic stepwise constant input current, variable 0
    real *pi_1;      	//!< presynaptic stepwise constant input current
    real *pi_syn_ex; 	//!< postsynaptic current for exc. inputs, variable 1
    real *pi_syn_in; 	//!< postsynaptic current for inh. inputs, variable 1
	real *pV_m;			// Membrane voltage in mV

	// internal param
	real *pP11ex;
	real *pP11in;
	real *pP22;
	real *pP21ex;
	real *pP21in;
	real *pP20;
	
	int *_fire_count;
};


size_t getIAFSize();
void *mallocIAF();
void *allocIAF(size_t num);
int allocIAFPara(void *pCPU, size_t num);
int freeIAF(void *pCPU);
int freeIAFPara(void *pCPU);
void updateIAF(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveIAF(void *pCPU, size_t num, const string &path);
void *loadIAF(size_t num, const string &path);
bool isEqualIAF(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);
int copyIAF(void *src, size_t s_off, void *dst, size_t d_off);
int logRateIAF(void *data, const char *name);
real * getVIAF(void *data);

void *cudaMallocIAF();
void *cudaAllocIAF(void *pCPU, size_t num);
void *cudaAllocIAFPara(void *pCPU, size_t num);
int cudaFreeIAF(void *pGPU);
int cudaFreeIAFPara(void *pGPU);
int cudaFetchIAF(void *pCPU, void *pGPU, size_t num);
int cudaIAFParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaIAFParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateIAF(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);
int cudaLogRateIAF(void *cpu, void *gpu, const char *name);
real * cudaGetVIAF(void *data);

int sendIAF(void *data, int dest, int tag, MPI_Comm comm);
void * recvIAF(int src, int tag, MPI_Comm comm);

#endif /* IAFDATA_H */
