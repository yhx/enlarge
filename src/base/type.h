/* This header file is writen by qp09
 * usually just for fun
 * Thu February 16 2017
 */
#ifndef TYPE_H
#define TYPE_H

#include "constant.h"

enum Type { 
	// Constant = 0, 
	// Poisson = 1,
	// Array = 1,
	// Decide,
	// FFT,
	// Mem,
	// Max,
	LIF = 0,
    // Izhikevich=1,
    // TraubMiles=2,
	// TJ,
	Static=1,
	// STDP,
	TYPESIZE,
	UNSET
}; 

enum SpikeType { Exc = 0, Inh = 1, SPIKETYPESIZE = 2};

enum SplitType { NeuronBalance = 0, RoundRobin = 1, GrpRR = 2, SynBestFit = 3, Metis = 4, BestFit = 5, Balanced, SynapseBalance};
#endif /* TYPE_H */

