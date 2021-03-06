/* This header file is writen by qp09
 * usually just for fun
 * Mon January 18 2016
 */
#ifndef GNETWORK_H
#define GNETWORK_H

#include "../utils/type.h"

#include "Connection.h"

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

	//Posize_ters to neurons
	void **ppNeurons;
	//Pointers to synapses
	void **ppSynapses;

	//Neuron to Synapse Connection
	Connection **ppConnection;

};


// init and free
// This func just set the content of ppConnection to NULL
GNetwork * allocGNetwork(size_t nTypeNum, size_t sTypeNum);
GNetwork * deepcopyGNetwork(GNetwork *net);
// TODO freeGNetwork
void freeGNetwork(GNetwork * network);

// Save and Load
int saveGNetwork(GNetwork *net, FILE *f);
GNetwork *loadGNetwork(FILE *f);
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

#endif /* GNETWORK_H */

