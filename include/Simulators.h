/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef SIMULATORS_H
#define SIMULATORS_H

#include "Neurons.h"
#include "Synapses.h"

//CPU SIM
#include "../src/sim/SingleThreadSimulator.h"
#include "../src/sim/MultiThreadSimulator.h"

//GPU SIM
#include "../src/sim/SingleGPUSimulator.h"
#include "../src/sim/MultiGPUSimulator.h"

//MPI SIM
#include "../src/sim/MultiNodeSimulator.h"
#include "../src/sim/MultiLevelSimulator.h"

// CPU + GPU + MPI
#include "../src/sim/HybridSimulator.h"

typedef SingleThreadSimulator STSim;
typedef SingleGPUSimulator SGSim;
typedef MultiGPUSimulator MGSim;
typedef MultiNodeSimulator MNSim;
typedef MultiLevelSimulator MLSim;
typedef MultiThreadSimulator MTSim;
typedef HybridSimulator HSim;

#endif /* SIMULATORS_H */

