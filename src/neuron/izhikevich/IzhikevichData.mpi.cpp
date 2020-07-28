#include "mpi.h"

#include "IzhikevichData.h"

int sendIzhikevich(void *data, int dest, int tag, MPI_Comm comm)
{
	return 0;
}

void * recvIzhikevich(int src, int tag, MPI_Comm comm)
{
	return NULL;
}
