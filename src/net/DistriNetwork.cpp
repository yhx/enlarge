/* This program is writen by qp09.
 * usually just for fun.
 * 四 三月 09 2017
 */

#include <stdlib.h>
#include <assert.h>

#include "DistriNetwork.h"

DistriNetwork* initDistriNet(int num, real dt)
{
	DistriNetwork * ret_net = (DistriNetwork*)malloc(sizeof(DistriNetwork) * num);

	for (int i=0; i<num; i++) {
		ret_net[i]._simCycle = 0;
		ret_net[i]._nodeIdx = i;
		ret_net[i]._nodeNum = num;

		ret_net[i]._dt = dt;

		ret_net[i]._network = NULL;
		ret_net[i]._crossnodeMap = NULL; 
	}

	return ret_net;
}

void freeDistriNet(DistriNetwork * net)
{
	int num = net->_nodeNum;
	for (int i=0; i<num; i++) {

		if (net[i]._crossnodeMap != NULL) {
			free(net[i]._crossnodeMap->_idx2index);
			free(net[i]._crossnodeMap->_crossnodeIndex2idx);
			free(net[i]._crossnodeMap); 
			net[i]._crossnodeMap = NULL;
		}

		//TODO free GNetwork
	}
	free(net);
}


int sendDistriNet(DistriNetwork *network, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(network, sizeof(DistriNetwork), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = sendGNetwork(network->_network, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = sendMap(network->_crossnodeMap, dest, tag+NET_TAG, comm);
	assert(ret == MPI_SUCCESS);
	return MPI_SUCCESS;
}

DistriNetwork *recvDistriNet(int src, int tag, MPI_Comm comm) 
{
	DistriNetwork *ret = (DistriNetwork*)malloc(sizeof(DistriNetwork));
	MPI_Status status;
	MPI_Recv(ret, sizeof(DistriNetwork), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	ret->_network = recvGNetwork(src, tag+1, comm);
	assert(ret->_network != NULL);
	ret->_crossnodeMap = recvMap(src, tag+NET_TAG, comm);
	assert(ret->_crossnodeMap != NULL);

	return ret;
}

