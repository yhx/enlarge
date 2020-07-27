
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
	CrossNodeMap *ret = (CrossNodeMap *)malloc(sizeof(CrossNodeMap));
	MPI_Status status;
	MPI_Recv(ret, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	ret->_idx2index = (int *)malloc(sizeof(int)*ret->_num);
	MPI_Recv(ret->_idx2index, ret->_num, MPI_INT, src, tag+1, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	ret->_crossnodeIndex2idx = (int *)malloc(sizeof(int)*(ret->_crossSize));
	MPI_Recv(ret->_crossnodeIndex2idx, ret->_crossSize, MPI_INT, src, tag+2, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	return ret;
}
