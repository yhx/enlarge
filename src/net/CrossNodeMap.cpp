
#include <stdlib.h>
#include <assert.h>

#include "CrossNodeMap.h"

int sendMap(CrossNodeMap *map_, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(map_, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_idx2index, map_->_num, MPI_INT, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_crossnodeIndex2idx, map_->_crossSize, MPI_INT, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

CrossNodeMap * recvMap(int src, int tag, MPI_Comm comm)
{
	CrossNodeMap *net = (CrossNodeMap *)malloc(sizeof(CrossNodeMap));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(net, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_idx2index = (int *)malloc(sizeof(int)*net->_num);
	ret = MPI_Recv(net->_idx2index, net->_num, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_crossnodeIndex2idx = (int *)malloc(sizeof(int)*(net->_crossSize));
	ret = MPI_Recv(net->_crossnodeIndex2idx, net->_crossSize, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	return net;
}
