
#include <assert.h>
#include "mpi.h"

#include "StaticData.h"

int sendStatic(void *data_, int dest, int tag, MPI_Comm comm)
{
	StaticData *data = (StaticData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pDst, data->num, MPI_INT, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pWeight, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvStatic(int src, int tag, MPI_Comm comm)
{
	StaticData *ret = (StaticData *)mallocStatic();
	MPI_Status status;
	MPI_Recv(&(ret->num), 1, MPI_INT, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	allocStaticPara(ret, ret->num);

	MPI_Recv(ret->pDst, ret->num, MPI_INT, src, tag+1, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	MPI_Recv(ret->pWeight, ret->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	return ret;
}
