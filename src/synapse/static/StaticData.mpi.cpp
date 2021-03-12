
#include <assert.h>
#include "mpi.h"

#include "StaticData.h"

int sendStatic(void *data_, int dest, int tag, MPI_Comm comm)
{
	StaticData *data = (StaticData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	// ret = MPI_Send(data->pDst, data->num, MPI_INT, dest, tag+1, comm);
	// assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pWeight, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvStatic(int src, int tag, MPI_Comm comm)
{
	StaticData *net = (StaticData *)mallocStatic();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocStaticPara(net, net->num);

	// ret = MPI_Recv(net->pDst, net->num, MPI_INT, src, tag+1, comm, &status);
	// assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pWeight, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}
