/* This header file is writen by qp09
 * usually just for fun
 * Mon January 18 2016
 */
#ifndef GNETWORK_H
#define GNETWORK_H

#include <string>

#include "../base/type.h"
#include "Connection.h"

using std::string;

struct GNetwork {
	//Numbers of types
	size_t nTypeNum;
	size_t sTypeNum;

	// Delay info moved size_to connection
	// size_t maxDelay;
	// size_t minDelay;

	//Type 
	Type * pNTypes;
	Type * pSTypes;

	//Index for each type
	size_t *pNeuronNums;
	size_t *pSynapseNums;

	size_t *bufferOffsets;

	//Posize_ters to neurons
	void **ppNeurons;
	//Pointers to synapses
	void **ppSynapses;

	//Neuron to Synapse Connection
	Connection **ppConnections;

};


// init and free
// This func just set the content of ppConnections to NULL
GNetwork * allocGNetwork(size_t nTypeNum, size_t sTypeNum);
GNetwork * deepcopyGNetwork(GNetwork *net);
void freeGNetwork(GNetwork * network);

// Save and Load
int saveGNetwork(GNetwork *net, const string &path);
GNetwork *loadGNetwork(const string &path);
bool compareGNetwork(GNetwork *n1, GNetwork *n2);

// Transfer GNetwork between CPU and GPU
// Only copy inside data arrays to GPU, the info data is left on CPU
GNetwork* copyGNetworkToGPU(GNetwork *);
int fetchGNetworkFromGPU(GNetwork *, GNetwork*);
int freeGNetworkGPU(GNetwork *);

// MPI
int copyGNetwork(GNetwork *dNet, GNetwork *sNet, int rank, int rankSize);
int sendGNetwork(GNetwork *network, int dst, int tag, MPI_Comm comm);
GNetwork * recvGNetwork(int src, int tag, MPI_Comm comm);


// Other utils
int printGNetwork(GNetwork *net, int rank = 0);

int pInfoGNetwork(GNetwork *net, const string &s = "");

#endif /* GNETWORK_H */

