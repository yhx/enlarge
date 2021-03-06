
#include <assert.h>
#include "mpi.h"

#include "PoissonData.h"

int sendPoisson(void *data_, int dest, int tag, MPI_Comm comm)
{
	PoissonData *data = (PoissonData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	// ret = MPI_Send(data->pDst, data->num, MPI_INT, dest, tag+1, comm);
	// assert(ret == MPI_SUCCESS);

	// eg: MPI_Send(send_buf_p,send_buf_sz,send_type,dest,send_tag,send_comm)
	ret = MPI_Send(data->pWeight, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pMean, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	// ret = MPI_Send(data->pState, data->num*sizeof(curandState), MPI_UNSIGNED_CHAR, dest, tag+2, comm);
	// assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvPoisson(int src, int tag, MPI_Comm comm)
{
	PoissonData *net = (PoissonData *)mallocPoisson();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocPoissonPara(net, net->num);

	// ret = MPI_Recv(net->pDst, net->num, MPI_INT, src, tag+1, comm, &status);
	// assert(ret==MPI_SUCCESS);

	// eg: MPI_Recv(recv_buf_p,recv_buf_sz,recv_type,src,recv_tag,recv_comm)
	ret = MPI_Recv(net->pWeight, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pMean, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret == MPI_SUCCESS);

	

	return net;
}
