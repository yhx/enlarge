
#ifndef DISTRINETWORK_H
#define DISTRINETWORK_H

#include "GNetwork.h"
#include "CrossNodeMap.h"

struct DistriNetwork {
	int _simCycle;
	int _nodeIdx;
	int _nodeNum;
	real _dt;
	GNetwork * _network;
	CrossNodeMap *_crossnodeMap;
	//CrossNodeData *_crossnodeData;
};

// Pointers inited to NULL, specific funcs in MultiNetwork will deal with these pointers later 
DistriNetwork* initDistriNet(int num, real dt=1e-4);

// Not NULL pointers are also freed.
DistriNetwork* freeDistriNet(int num);

int sendDistriNet(DistriNetwork *network, int dest, int tag, MPI_Comm comm);
DistriNetwork * recvDistriNet(int src, int tag, MPI_Comm comm);

#endif // DISTRINETWORK_H
