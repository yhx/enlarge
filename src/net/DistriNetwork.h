
#ifndef DISTRINETWORK_H
#define DISTRINETWORK_H

#include <string>

#include "GNetwork.h"
#include "../msg_utils/CrossNodeMap.h"
#include "../msg_utils/CrossNodeData.h"
#include "../msg_utils/CrossThreadData.h"

using std::string;

struct DistriNetwork {
	int _simCycle;                  // 仿真时间步
	int _nodeIdx;                   // 
	int _nodeNum;
	real _dt;
	GNetwork * _network;
	CrossNodeMap *_crossnodeMap;
};

// Pointers inited to NULL, specific funcs in MultiNetwork will deal with these pointers later 
DistriNetwork* initDistriNet(int num, real dt=1e-4);

// Not NULL pointers are also freed.
DistriNetwork* freeDistriNet(int num);

int sendDistriNet(DistriNetwork *network, int dest, int tag, MPI_Comm comm);
DistriNetwork * recvDistriNet(int src, int tag, MPI_Comm comm);

int saveDistriNet(DistriNetwork *net, const string &path);
DistriNetwork * loadDistriNet(const string &path);
bool compareDistriNet(DistriNetwork *n1, DistriNetwork *n2);

#endif // DISTRINETWORK_H
