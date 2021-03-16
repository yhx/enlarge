/* This program is writen by qp09.
 * usually just for fun.
 * 四 三月 09 2017
 */

#include <stdlib.h>
#include <assert.h>

#include "../utils/helper_c.h"
#include "GNetwork.h"
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
	DistriNetwork *net = (DistriNetwork*)malloc(sizeof(DistriNetwork));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(net, sizeof(DistriNetwork), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_network = recvGNetwork(src, tag+1, comm);
	assert(net->_network != NULL);
	net->_crossnodeMap = recvMap(src, tag+NET_TAG, comm);
	assert(net->_crossnodeMap != NULL);

	return net;
}

int saveDistriNet(DistriNetwork *net, FILE * f)
{
	fwrite_c(&(net->_simCycle), sizeof(int), 1, f);
	fwrite_c(&(net->_nodeIdx), sizeof(int), 1, f);
	fwrite_c(&(net->_nodeNum), sizeof(int), 1, f);
	fwrite_c(&(net->_dt), sizeof(real), 1, f);
	saveGNetwork(net->_network, f);
	saveCNM(net->_crossnodeMap, f);

	return 0;
}

DistriNetwork * loadDistriNet(FILE *f) {
	DistriNetwork *net = (DistriNetwork*)malloc(sizeof(DistriNetwork));
	fread_c(&(net->_simCycle), sizeof(int), 1, f);
	fread_c(&(net->_nodeIdx), sizeof(int), 1, f);
	fread_c(&(net->_nodeNum), sizeof(int), 1, f);
	fread_c(&(net->_dt), sizeof(real), 1, f);
	net->_network = loadGNetwork(f);
	net->_crossnodeMap = loadCNM(f);
	return net;
}

bool compareDistriNet(DistriNetwork *n1, DistriNetwork *n2)
{
	bool equal = true;
	equal = equal && (n1->_simCycle == n2->_simCycle);
	equal = equal && (n1->_nodeIdx == n2->_nodeIdx);
	equal = equal && (n1->_nodeNum == n2->_nodeNum);
	equal = equal && (n1->_dt == n2->_dt);
	equal = compareGNetwork(n1->_network, n2->_network) && equal;
	equal = compareCNM(n1->_crossnodeMap, n2->_crossnodeMap) && equal;

	return equal;
}
