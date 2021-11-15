#include <assert.h>
#include "mpi.h"

#include "IAFData.h"

int sendIAF(void *data_, int dest, int tag, MPI_Comm comm)
{
	IAFData * data = (IAFData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pRefracTime, data->num, MPI_INT, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pRefracStep, data->num, MPI_INT, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	// model param
	ret = MPI_Send(data->pE_L, data->num, MPI_U_REAL, dest, tag+17, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pI_e, data->num, MPI_U_REAL, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pTheta, data->num, MPI_U_REAL, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pV_reset, data->num, MPI_U_REAL, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);

	// state param
	ret = MPI_Send(data->pi_0, data->num, MPI_U_REAL, dest, tag+6, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pi_1, data->num, MPI_U_REAL, dest, tag+7, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pi_syn_ex, data->num, MPI_U_REAL, dest, tag+8, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pi_syn_in, data->num, MPI_U_REAL, dest, tag+9, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pV_m, data->num, MPI_U_REAL, dest, tag+10, comm);
	assert(ret == MPI_SUCCESS);

	// internal param
	ret = MPI_Send(data->pP11ex, data->num, MPI_U_REAL, dest, tag+11, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pP11in, data->num, MPI_U_REAL, dest, tag+12, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pP22, data->num, MPI_U_REAL, dest, tag+13, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pP21ex, data->num, MPI_U_REAL, dest, tag+14, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pP21in, data->num, MPI_U_REAL, dest, tag+15, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pP20, data->num, MPI_U_REAL, dest, tag+16, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvIAF(int src, int tag, MPI_Comm comm)
{
	IAFData *net = (IAFData *)mallocIAF();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocIAFPara(net, net->num);

	ret = MPI_Recv(net->pRefracTime, net->num, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pRefracStep, net->num, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pE_L, net->num, MPI_U_REAL, src, tag+17, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pI_e, net->num, MPI_U_REAL, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pTheta, net->num, MPI_U_REAL, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pV_reset, net->num, MPI_U_REAL, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pi_0, net->num, MPI_U_REAL, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pi_1, net->num, MPI_U_REAL, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pi_syn_ex, net->num, MPI_U_REAL, src, tag+8, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pi_syn_in, net->num, MPI_U_REAL, src, tag+9, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pV_m, net->num, MPI_U_REAL, src, tag+10, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pP11ex, net->num, MPI_U_REAL, src, tag+11, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pP11in, net->num, MPI_U_REAL, src, tag+12, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pP22, net->num, MPI_U_REAL, src, tag+13, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pP21ex, net->num, MPI_U_REAL, src, tag+14, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pP21in, net->num, MPI_U_REAL, src, tag+15, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pP20, net->num, MPI_U_REAL, src, tag+16, comm, &status);
	assert(ret==MPI_SUCCESS);
	
	return net;
}
